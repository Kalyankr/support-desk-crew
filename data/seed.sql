-- Mock account database for Support Desk Crew.
-- Fully rebuilt by `support_desk.data.db.build()`. All dates are relative to
-- config.AS_OF = 2026-10-01 so policy windows are deterministic.

PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    tier  TEXT NOT NULL CHECK (tier IN ('standard', 'plus'))
);

CREATE TABLE orders (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    placed_at   TEXT NOT NULL,
    item        TEXT NOT NULL,
    total       REAL NOT NULL CHECK (total > 0),
    status      TEXT NOT NULL CHECK (status IN ('placed', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE payments (
    id         TEXT PRIMARY KEY,
    order_id   TEXT NOT NULL REFERENCES orders(id),
    amount     REAL NOT NULL CHECK (amount > 0),
    charged_at TEXT NOT NULL,
    refunded   INTEGER NOT NULL DEFAULT 0 CHECK (refunded IN (0, 1))
);

CREATE TABLE shipments (
    id           TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL REFERENCES orders(id),
    carrier      TEXT NOT NULL,
    tracking     TEXT NOT NULL,
    last_scan_at TEXT
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_payments_order  ON payments(order_id);
CREATE INDEX idx_shipments_order ON shipments(order_id);

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Customers
-- ---------------------------------------------------------------------------
INSERT INTO customers (id, name, email, tier) VALUES
  ('C-001', 'Ada Mercer',        'ada.mercer@example.com',        'standard'),
  ('C-002', 'Ben Okafor',        'ben.okafor@example.com',        'plus'),
  ('C-003', 'Cara Lindqvist',    'cara.lindqvist@example.com',    'standard'),
  ('C-004', 'Dev Raghunathan',   'dev.raghunathan@example.com',   'plus'),
  ('C-005', 'Elena Marchetti',   'elena.marchetti@example.com',   'standard'),
  ('C-006', 'Farid Haddad',      'farid.haddad@example.com',      'standard'),
  ('C-007', 'Grace Nakamura',    'grace.nakamura@example.com',    'plus'),
  ('C-008', 'Hugo Delacroix',    'hugo.delacroix@example.com',    'standard'),
  ('C-009', 'Iris Abiodun',      'iris.abiodun@example.com',      'standard'),
  ('C-010', 'Jonas Wexler',      'jonas.wexler@example.com',      'plus'),
  ('C-011', 'Kiara Sandoval',    'kiara.sandoval@example.com',    'standard'),
  ('C-012', 'Liam O''Donnell',   'liam.odonnell@example.com',     'standard');

-- ---------------------------------------------------------------------------
-- Orders. Every refund/warranty/shipping edge case is represented here.
-- ---------------------------------------------------------------------------
INSERT INTO orders (id, customer_id, placed_at, item, total, status) VALUES
  ('L-10401', 'C-001', '2026-09-24', 'Lumen desk lamp',                   249.00, 'delivered'),  --   7d standard, refundable
  ('L-10402', 'C-002', '2026-08-20', 'Lumen desk lamp',                   249.00, 'delivered'),  --  42d plus, inside 60d
  ('L-10403', 'C-003', '2026-08-12', 'Lumen desk lamp',                   249.00, 'delivered'),  --  50d standard, outside 30d
  ('L-10404', 'C-004', '2026-07-15', 'Lumen desk lamp x2',                498.00, 'delivered'),  --  78d plus, outside 60d
  ('L-10405', 'C-005', '2026-09-18', 'Lumen desk lamp + expedited',       274.00, 'delivered'),  --  13d, cosmetic damage case
  ('L-10406', 'C-006', '2026-09-29', 'Lumen desk lamp',                   249.00, 'shipped'),    --   2d, in transit
  ('L-10407', 'C-007', '2026-09-10', 'Lumen desk lamp',                   249.00, 'delivered'),  --  21d plus, ALREADY refunded
  ('L-10408', 'C-008', '2026-09-27', 'Lumen desk lamp',                   249.00, 'placed'),     --   4d, not yet dispatched
  ('L-10409', 'C-009', '2026-09-05', 'Lumen desk lamp',                   249.00, 'shipped'),    --  26d, tracking stale 20d
  ('L-10410', 'C-010', '2026-09-21', 'Lumen desk lamp x2',                498.00, 'delivered'),  --  10d plus
  ('L-10411', 'C-011', '2026-09-30', 'Lumen Arm Mount',                    79.00, 'shipped'),    --   1d accessory
  ('L-10412', 'C-012', '2026-09-15', 'Lumen desk lamp',                   249.00, 'delivered'),  --  16d, flicker defect
  ('L-10413', 'C-001', '2026-06-02', 'Lumen desk lamp',                   249.00, 'delivered'),  -- 121d, in warranty
  ('L-10414', 'C-002', '2026-09-26', 'Lumen Travel Case',                  49.00, 'delivered'),  --   5d, sub-$100 refund
  ('L-10415', 'C-003', '2026-09-28', 'Lumen desk lamp',                   249.00, 'cancelled'),  --   cancelled, never charged
  ('L-10416', 'C-004', '2026-09-12', 'Lumen desk lamp',                   249.00, 'delivered'),  --  19d plus, dimmer fault
  ('L-10417', 'C-005', '2026-09-23', 'Lumen Arm Mount',                    79.00, 'delivered'),  --   8d, sub-$100 refund
  ('L-10418', 'C-006', '2026-09-08', 'Lumen desk lamp',                    249.00, 'shipped'),   --  23d, tracking stale 9d
  ('L-10419', 'C-007', '2026-05-20', 'Lumen desk lamp',                   249.00, 'delivered'),  -- 134d plus, far outside
  ('L-10420', 'C-008', '2026-09-19', 'Lumen desk lamp x2',                498.00, 'delivered'),  --  12d standard, DOUBLE charge
  ('L-10421', 'C-009', '2026-09-30', 'Lumen desk lamp',                   249.00, 'placed'),     --   1d
  ('L-10422', 'C-010', '2026-09-17', 'Lumen desk lamp',                   249.00, 'delivered'),  --  14d plus, DOUBLE charge
  ('L-10423', 'C-011', '2026-08-28', 'Lumen desk lamp',                   249.00, 'delivered'),  --  34d standard, 4d past window
  ('L-10424', 'C-012', '2026-09-25', 'Lumen desk lamp + expedited',       274.00, 'shipped'),    --   6d, expedited SLA missed
  ('L-10425', 'C-003', '2026-09-11', 'Lumen Travel Case',                  49.00, 'delivered');  --  20d, marked delivered, not received

-- ---------------------------------------------------------------------------
-- Payments. L-10415 is cancelled and has none. L-10420 and L-10422 were charged twice.
-- ---------------------------------------------------------------------------
INSERT INTO payments (id, order_id, amount, charged_at, refunded) VALUES
  ('P-20001', 'L-10401', 249.00, '2026-09-24', 0),
  ('P-20002', 'L-10402', 249.00, '2026-08-20', 0),
  ('P-20003', 'L-10403', 249.00, '2026-08-12', 0),
  ('P-20004', 'L-10404', 498.00, '2026-07-15', 0),
  ('P-20005', 'L-10405', 274.00, '2026-09-18', 0),
  ('P-20006', 'L-10406', 249.00, '2026-09-29', 0),
  ('P-20007', 'L-10407', 249.00, '2026-09-10', 1),   -- already refunded
  ('P-20008', 'L-10408', 249.00, '2026-09-27', 0),
  ('P-20009', 'L-10409', 249.00, '2026-09-05', 0),
  ('P-20010', 'L-10410', 498.00, '2026-09-21', 0),
  ('P-20011', 'L-10411',  79.00, '2026-09-30', 0),
  ('P-20012', 'L-10412', 249.00, '2026-09-15', 0),
  ('P-20013', 'L-10413', 249.00, '2026-06-02', 0),
  ('P-20014', 'L-10414',  49.00, '2026-09-26', 0),
  ('P-20015', 'L-10416', 249.00, '2026-09-12', 0),
  ('P-20016', 'L-10417',  79.00, '2026-09-23', 0),
  ('P-20017', 'L-10418', 249.00, '2026-09-08', 0),
  ('P-20018', 'L-10419', 249.00, '2026-05-20', 0),
  ('P-20019', 'L-10420', 498.00, '2026-09-19', 0),
  ('P-20020', 'L-10420', 498.00, '2026-09-19', 0),   -- duplicate charge
  ('P-20021', 'L-10421', 249.00, '2026-09-30', 0),
  ('P-20022', 'L-10422', 249.00, '2026-09-17', 0),
  ('P-20023', 'L-10422', 249.00, '2026-09-18', 0),   -- duplicate charge
  ('P-20024', 'L-10423', 249.00, '2026-08-28', 0),
  ('P-20025', 'L-10424', 274.00, '2026-09-25', 0),
  ('P-20026', 'L-10425',  49.00, '2026-09-11', 0);

-- ---------------------------------------------------------------------------
-- Shipments. Only for orders that left the warehouse.
-- ---------------------------------------------------------------------------
INSERT INTO shipments (id, order_id, carrier, tracking, last_scan_at) VALUES
  ('S-30001', 'L-10401', 'Northline', 'NL7741020391', '2026-09-27'),
  ('S-30002', 'L-10402', 'Northline', 'NL7738810225', '2026-08-24'),
  ('S-30003', 'L-10403', 'Veloce',    'VE5590113087', '2026-08-16'),
  ('S-30004', 'L-10404', 'Veloce',    'VE5582004411', '2026-07-19'),
  ('S-30005', 'L-10405', 'Veloce',    'VE5596620118', '2026-09-20'),
  ('S-30006', 'L-10406', 'Northline', 'NL7744501882', '2026-09-30'),
  ('S-30007', 'L-10407', 'Postaro',   'PO9911274500', '2026-09-14'),
  ('S-30008', 'L-10409', 'Postaro',   'PO9908831276', '2026-09-11'),  -- stale 20d -> lost
  ('S-30009', 'L-10410', 'Northline', 'NL7742990014', '2026-09-25'),
  ('S-30010', 'L-10411', 'Postaro',   'PO9913360842', NULL),          -- no scan yet
  ('S-30011', 'L-10412', 'Northline', 'NL7740118823', '2026-09-19'),
  ('S-30012', 'L-10413', 'Veloce',    'VE5571009934', '2026-06-06'),
  ('S-30013', 'L-10414', 'Postaro',   'PO9912880671', '2026-09-29'),
  ('S-30014', 'L-10416', 'Northline', 'NL7739770456', '2026-09-16'),
  ('S-30015', 'L-10417', 'Veloce',    'VE5597741203', '2026-09-26'),
  ('S-30016', 'L-10418', 'Postaro',   'PO9909920318', '2026-09-22'),  -- stale 9d -> resend
  ('S-30017', 'L-10419', 'Veloce',    'VE5563300872', '2026-05-24'),
  ('S-30018', 'L-10420', 'Northline', 'NL7741880299', '2026-09-23'),
  ('S-30019', 'L-10422', 'Northline', 'NL7740990137', '2026-09-21'),
  ('S-30020', 'L-10423', 'Veloce',    'VE5588120644', '2026-09-01'),
  ('S-30021', 'L-10424', 'Veloce',    'VE5598830225', '2026-09-26'),  -- expedited, still in transit
  ('S-30022', 'L-10425', 'Postaro',   'PO9910045529', '2026-09-14');  -- scanned delivered
