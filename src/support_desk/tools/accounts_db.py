"""Parameterised account-database tools.

These are the only way anything — agent or human code — touches the accounts database.
Every query uses `?` placeholders; no caller-supplied string is ever interpolated into SQL.
The model never writes SQL, and never will: it can only call these typed functions.
"""

from __future__ import annotations

import sqlite3
from datetime import date

from support_desk.data.db import connect
from support_desk.schemas import CustomerRecord, OrderRecord, PaymentRecord, ShipmentRecord


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _customer(row: sqlite3.Row) -> CustomerRecord:
    return CustomerRecord(id=row["id"], name=row["name"], email=row["email"], tier=row["tier"])


def _order(row: sqlite3.Row) -> OrderRecord:
    return OrderRecord(
        id=row["id"],
        customer_id=row["customer_id"],
        placed_at=_date(row["placed_at"]),
        item=row["item"],
        total=row["total"],
        status=row["status"],
    )


def _payment(row: sqlite3.Row) -> PaymentRecord:
    return PaymentRecord(
        id=row["id"],
        order_id=row["order_id"],
        amount=row["amount"],
        charged_at=_date(row["charged_at"]),
        refunded=bool(row["refunded"]),
    )


def _shipment(row: sqlite3.Row) -> ShipmentRecord:
    return ShipmentRecord(
        id=row["id"],
        order_id=row["order_id"],
        carrier=row["carrier"],
        tracking=row["tracking"],
        last_scan_at=_date(row["last_scan_at"]) if row["last_scan_at"] else None,
    )


def get_customer(email: str, conn: sqlite3.Connection | None = None) -> CustomerRecord | None:
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute(
            "SELECT id, name, email, tier FROM customers WHERE email = ?", (email,)
        ).fetchone()
        return _customer(row) if row else None
    finally:
        if own:
            conn.close()


def get_order(order_id: str, conn: sqlite3.Connection | None = None) -> OrderRecord | None:
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute(
            "SELECT id, customer_id, placed_at, item, total, status FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
        return _order(row) if row else None
    finally:
        if own:
            conn.close()


def list_recent_orders(
    customer_id: str, limit: int = 5, conn: sqlite3.Connection | None = None
) -> list[OrderRecord]:
    own = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            "SELECT id, customer_id, placed_at, item, total, status FROM orders "
            "WHERE customer_id = ? ORDER BY placed_at DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
        return [_order(row) for row in rows]
    finally:
        if own:
            conn.close()


def get_payment(order_id: str, conn: sqlite3.Connection | None = None) -> PaymentRecord | None:
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute(
            "SELECT id, order_id, amount, charged_at, refunded FROM payments WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return _payment(row) if row else None
    finally:
        if own:
            conn.close()


def get_shipment(order_id: str, conn: sqlite3.Connection | None = None) -> ShipmentRecord | None:
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute(
            "SELECT id, order_id, carrier, tracking, last_scan_at FROM shipments "
            "WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return _shipment(row) if row else None
    finally:
        if own:
            conn.close()
