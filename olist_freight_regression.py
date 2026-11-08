"""Module 3 / Q2 - Reproducible Olist freight linear regression.

Cold run from the same folder as olist_freight_full.csv:
    python -m pip install pandas numpy statsmodels
    python olist_freight_regression.py

To repeat the SQL pull against PostgreSQL (requires psycopg2-binary too):
    export DATABASE_URL='postgresql://...'
    python olist_freight_regression.py --source db

The fitted sample is single-item orders only. HC3 robust standard errors;
original freight units R$, weight kg, dimensions cm. No coefficients invented.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Exact SQL used for the DBeaver export. PostgreSQL schema is 'olist'.
SQL = """
WITH single_item_orders AS (
    SELECT order_id
    FROM olist.order_items
    GROUP BY order_id
    HAVING COUNT(*) = 1
)
SELECT
    oi.order_id,
    oi.freight_value AS freight_value_brl,
    p.product_weight_g / 1000.0 AS weight_kg,
    p.product_length_cm AS length_cm,
    p.product_height_cm AS height_cm,
    p.product_width_cm AS width_cm,
    c.customer_state
FROM single_item_orders AS so
JOIN olist.order_items AS oi ON so.order_id = oi.order_id
JOIN olist.products AS p ON oi.product_id = p.product_id
JOIN olist.orders AS o ON oi.order_id = o.order_id
JOIN olist.customers AS c ON o.customer_id = c.customer_id
ORDER BY oi.order_id;
"""

NUMERIC = ["freight_value_brl", "weight_kg", "length_cm", "height_cm", "width_cm"]
REQUIRED = ["order_id", *NUMERIC, "customer_state"]
KEY_STATES = {"SP", "RJ", "MG", "PR", "RS"}
FORMULA = (
    'freight_value_brl ~ weight_kg + length_cm + height_cm + width_cm '
    '+ C(customer_state_group, Treatment(reference="SP"))'
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Put olist_freight_full.csv next to this script "
            "(included in the submission package), or use --source db."
        )
    return pd.read_csv(path, sep="\t", dtype={"order_id": "string", "customer_state": "string"})


def load_database() -> pd.DataFrame:
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError('For --source db, install "psycopg2-binary".') from exc
    url = os.environ.get("DATABASE_URL")
    if url:
        conn = psycopg2.connect(url)
    else:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "BAN5003"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
    try:
        return pd.read_sql_query(SQL, conn)
    finally:
        conn.close()


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    x = raw[REQUIRED].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["customer_state"] = x["customer_state"].astype("string").str.strip().str.upper()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    x = x[(x["freight_value_brl"] >= 0) & (x["weight_kg"] > 0)]
    x = x[(x[["length_cm", "height_cm", "width_cm"]] > 0).all(axis=1)]
    x = x[x["customer_state"].str.fullmatch(r"[A-Z]{2}", na=False)]
    x = x.drop_duplicates(subset=["order_id"])
    x["customer_state_group"] = x["customer_state"].where(
        x["customer_state"].isin(KEY_STATES), "OTHER"
    ).astype(str)
    if not (x["customer_state_group"] == "SP").any():
        raise ValueError("No São Paulo (SP) records for the chosen reference group.")
    if len(x) < 50:
        raise ValueError("Too few valid orders for this regression.")
    return x, {"sql_rows": len(raw), "excluded": len(raw) - len(x), "fitted": len(x)}


def business_label(term: str) -> str:
    names = {
        "Intercept": "Intercept (SP, zero size and weight)",
        "weight_kg": "Product weight (per 1 kg)",
        "length_cm": "Product length (per 1 cm)",
        "height_cm": "Product height (per 1 cm)",
        "width_cm": "Product width (per 1 cm)",
    }
    if term in names:
        return names[term]
    for abbr, name in [
        ("MG", "Minas Gerais"), ("OTHER", "Other states combined"),
        ("PR", "Paraná"), ("RJ", "Rio de Janeiro"),
        ("RS", "Rio Grande do Sul")
    ]:
        if f"[T.{abbr}]" in term:
            return f"{name} vs São Paulo"
    return term


def run(raw: pd.DataFrame, output_dir: Path) -> None:
    data, info = clean(raw)
    fit = smf.ols(FORMULA, data=data).fit(cov_type="HC3")
    ci = fit.conf_int(alpha=0.05)
    table = pd.DataFrame({
        "term": fit.params.index,
        "predictor": [business_label(t) for t in fit.params.index],
        "estimate_brl": fit.params.to_numpy(),
        "std_error_brl_hc3": fit.bse.to_numpy(),
        "ci_95_low_brl": ci.iloc[:, 0].to_numpy(),
        "ci_95_high_brl": ci.iloc[:, 1].to_numpy(),
        "p_value": fit.pvalues.to_numpy(),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "coefficient_table.csv", index=False, float_format="%.8f")
    with (output_dir / "model_results.txt").open("w", encoding="utf-8") as handle:
        handle.write("MODULE 3 / Q2: OLIST FREIGHT REGRESSION\n")
        handle.write(f"Rows returned by SQL/CSV: {info['sql_rows']:,}\n")
        handle.write(f"Rows excluded: {info['excluded']:,}\n")
        handle.write(f"Rows used in regression: {info['fitted']:,}\n")
        handle.write(f"R-squared: {fit.rsquared:.6f}\n")
        handle.write(f"Adjusted R-squared: {fit.rsquared_adj:.6f}\n")
        handle.write("OLS with HC3 robust standard errors. SP is the state reference.\n")
        handle.write("Weight is kg; dimensions cm; freight BRL.\n\n")
        handle.write(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        handle.write("\n\nMODEL SUMMARY\n" + fit.summary().as_text() + "\n")
    print(f"Raw orders: {info['sql_rows']:,}; excluded: {info['excluded']:,}; fitted: {info['fitted']:,}")
    print(f"R-squared: {fit.rsquared:.6f}")
    print(table[["predictor", "estimate_brl", "std_error_brl_hc3",
                 "ci_95_low_brl", "ci_95_high_brl"]].to_string(
                     index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote coefficient_table.csv and model_results.txt to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Module 3 Olist Q2 regression")
    parser.add_argument("--source", choices=["csv", "db"], default="csv")
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("olist_freight_full.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("regression_outputs"))
    args = parser.parse_args()
    raw = load_database() if args.source == "db" else load_csv(args.csv)
    run(raw, args.output_dir)


if __name__ == "__main__":
    main()