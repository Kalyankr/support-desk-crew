"""Account agent: resolves the customer/order/payment/shipment facts a ticket refers to.

Order ids follow a fixed, unambiguous format (L-#####), so this is deterministic regex
extraction plus parameterised tool calls — not a model call. No prompt extracts a fixed-format
id more reliably than a regex, and skipping the LLM here keeps this phase free, fast, and
fully testable without live API access. (Contrast with Phase 1: classifying free text into
a category genuinely needs judgment; finding "L-10422" in a string does not.)

Three failure modes are handled explicitly, per the ticket's own words:
- not_found   — no customer, or no order, matches.
- ambiguous   — the ticket mentions more than one order id, or an order that belongs to a
                different customer than the one who filed the ticket.
- tool_error  — the database raised; never let that crash the pipeline.
"""

from __future__ import annotations

import re
import sqlite3

from support_desk.schemas import AccountLookup
from support_desk.tools import accounts_db

_ORDER_ID = re.compile(r"\bL-\d{5}\b")


def account_lookup(
    ticket_text: str, customer_email: str, conn: sqlite3.Connection | None = None
) -> AccountLookup:
    try:
        customer = accounts_db.get_customer(customer_email, conn=conn)
    except sqlite3.Error as exc:
        return AccountLookup(status="tool_error", detail=str(exc))

    if customer is None:
        return AccountLookup(
            status="not_found", detail=f"no customer with email {customer_email!r}"
        )

    order_ids = sorted(set(_ORDER_ID.findall(ticket_text)))
    if len(order_ids) > 1:
        return AccountLookup(
            customer=customer,
            status="ambiguous",
            detail=f"ticket mentions multiple order ids: {order_ids}",
        )

    if not order_ids:
        return AccountLookup(customer=customer, status="found", detail="no order id in ticket text")

    order_id = order_ids[0]
    try:
        order = accounts_db.get_order(order_id, conn=conn)
    except sqlite3.Error as exc:
        return AccountLookup(customer=customer, status="tool_error", detail=str(exc))

    if order is None:
        return AccountLookup(customer=customer, status="not_found", detail=f"no order {order_id}")

    if order.customer_id != customer.id:
        return AccountLookup(
            customer=customer,
            order=order,
            status="ambiguous",
            detail=f"order {order_id} belongs to a different customer",
        )

    try:
        payment = accounts_db.get_payment(order_id, conn=conn)
        shipment = accounts_db.get_shipment(order_id, conn=conn)
    except sqlite3.Error as exc:
        return AccountLookup(customer=customer, order=order, status="tool_error", detail=str(exc))

    return AccountLookup(
        customer=customer, order=order, payment=payment, shipment=shipment, status="found"
    )
