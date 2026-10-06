"""Account agent: resolves the customer/order/payment/shipment facts a ticket refers to.

Two paths, one output contract:

- `account_lookup_llm` — the real tool-calling loop. The model gets tool schemas and decides
  what to call; we execute, feed results back as ToolMessages, and repeat until it answers
  or hits `max_steps`. This is Phase 2's learning goal: a model reaching outside itself.
- `account_lookup` — deterministic regex plus direct tool calls, no model. Serves as the
  fallback when the loop errors or exhausts its step budget.

The *decision* is the model's; the *resolution* is code's. `_resolve` performs the lookups
and the ownership check, so no prompt can talk its way into another customer's order.

Failure modes are explicit in `status`:
- no_order_referenced — ticket legitimately mentions no order.
- not_found           — no such customer, or no such order.
- ambiguous           — several order ids, or an order owned by a different customer.
- tool_error          — the database raised; never crash the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from support_desk import config
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import AccountLookup, CustomerRecord
from support_desk.tools import accounts_db

logger = logging.getLogger(__name__)

_ORDER_ID = re.compile(r"\bL-\d{5}\b")
_UNPARSED = object()


# --- Tool schemas the model sees ---------------------------------------------


class GetOrder(BaseModel):
    """Look up a single order by its id. Order ids look like L-10422."""

    order_id: str = Field(description="The order id, format L-#####")


class ListRecentOrders(BaseModel):
    """List a customer's recent orders, newest first. Use when the ticket names no order id."""

    customer_id: str = Field(description="The customer id, format C-###")
    limit: int = Field(default=5, ge=1, le=20)


TOOL_SCHEMAS: list[Any] = [GetOrder, ListRecentOrders]

_SYSTEM_PROMPT = """\
You identify which order a customer support ticket is about.

You have tools. Use them — never guess an order id, and never invent one that a tool did
not return. If the ticket names an order id, verify it with GetOrder. If it names none,
call ListRecentOrders and pick the order the ticket describes.

When you are done, reply with ONLY this JSON, no prose and no markdown fences:
{"order_id": "L-10422"}   or   {"order_id": null}

Use null if the ticket references no order, or if the order it names does not exist.
"""


def _run_tool(name: str, args: dict[str, Any], conn: sqlite3.Connection | None) -> str:
    """Execute one model-requested tool call, rendered as JSON for the model to read back."""
    if name == "GetOrder":
        order = accounts_db.get_order(args["order_id"], conn=conn)
        return order.model_dump_json() if order else json.dumps({"error": "no such order"})
    if name == "ListRecentOrders":
        orders = accounts_db.list_recent_orders(
            args["customer_id"], limit=args.get("limit", 5), conn=conn
        )
        return json.dumps([json.loads(o.model_dump_json()) for o in orders])
    return json.dumps({"error": f"unknown tool {name}"})


# --- Shared resolution: deterministic, not negotiable by prompt ---------------


def _resolve(
    customer: CustomerRecord, order_id: str | None, conn: sqlite3.Connection | None
) -> AccountLookup:
    if order_id is None:
        return AccountLookup(
            customer=customer, status="no_order_referenced", detail="ticket names no order"
        )

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
        payments = accounts_db.list_payments(order_id, conn=conn)
        shipment = accounts_db.get_shipment(order_id, conn=conn)
    except sqlite3.Error as exc:
        return AccountLookup(customer=customer, order=order, status="tool_error", detail=str(exc))

    return AccountLookup(
        customer=customer, order=order, payments=payments, shipment=shipment, status="found"
    )


def _load_customer(
    customer_email: str, conn: sqlite3.Connection | None
) -> tuple[CustomerRecord | None, AccountLookup | None]:
    try:
        customer = accounts_db.get_customer(customer_email, conn=conn)
    except sqlite3.Error as exc:
        return None, AccountLookup(status="tool_error", detail=str(exc))
    if customer is None:
        return None, AccountLookup(
            status="not_found", detail=f"no customer with email {customer_email!r}"
        )
    return customer, None


# --- Deterministic path -------------------------------------------------------


def account_lookup(
    ticket_text: str, customer_email: str, conn: sqlite3.Connection | None = None
) -> AccountLookup:
    """Regex extraction, no model. Also the fallback when the tool loop cannot finish."""
    customer, failure = _load_customer(customer_email, conn)
    if failure is not None:
        return failure
    assert customer is not None

    order_ids = sorted(set(_ORDER_ID.findall(ticket_text)))
    if len(order_ids) > 1:
        return AccountLookup(
            customer=customer,
            status="ambiguous",
            detail=f"ticket mentions multiple order ids: {order_ids}",
        )
    return _resolve(customer, order_ids[0] if order_ids else None, conn)


# --- Tool-calling path --------------------------------------------------------


class _ToolCallingModel(Protocol):
    def bind_tools(self, tools: list[Any]) -> Any: ...
    def invoke(self, messages: list[Any]) -> Any: ...


def _parse_answer(response: AIMessage) -> Any:
    content = response.content if isinstance(response.content, str) else str(response.content)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return _UNPARSED
    try:
        order_id = json.loads(match.group(0)).get("order_id")
    except json.JSONDecodeError:
        return _UNPARSED
    return order_id if isinstance(order_id, str) or order_id is None else _UNPARSED


def account_lookup_llm(
    ticket_text: str,
    customer_email: str,
    llm: _ToolCallingModel,
    conn: sqlite3.Connection | None = None,
    max_steps: int = 4,
    meter: BudgetMeter | None = None,
) -> AccountLookup:
    """Let the model choose which tools to call, then resolve its answer deterministically."""
    customer, failure = _load_customer(customer_email, conn)
    if failure is not None:
        return failure
    assert customer is not None

    model = llm.bind_tools(TOOL_SCHEMAS)
    messages: list[Any] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"customer_id={customer.id}\n\nticket:\n{ticket_text}"),
    ]

    for _step in range(max_steps):
        try:
            response = model.invoke(messages)
        except Exception as exc:
            logger.warning("account tool loop failed, falling back to regex: %s", exc)
            return account_lookup(ticket_text, customer_email, conn)

        if meter is not None:
            usage = getattr(response, "usage_metadata", None) or {}
            meter.record(
                agent="account",
                model=config.ACCOUNT_MODEL,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            )

        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            order_id = _parse_answer(response)
            if order_id is _UNPARSED:
                break
            return _resolve(customer, order_id, conn)

        for call in tool_calls:
            try:
                result = _run_tool(call["name"], call.get("args", {}), conn)
            except sqlite3.Error as exc:
                return AccountLookup(customer=customer, status="tool_error", detail=str(exc))
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    logger.warning("account tool loop did not produce an answer, falling back to regex")
    return account_lookup(ticket_text, customer_email, conn)
