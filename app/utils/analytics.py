"""
EMS Analytics Engine
====================
All KPI computation and chart-data serialisation lives here.
Routes call these functions; they return plain Python dicts/lists
that Flask can JSON-serialise directly.

Design rules
------------
- No rounding of raw data – only display values are rounded.
- Division-by-zero is always guarded.
- Functions return {} / [] on empty input rather than raising.
"""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd

from app.utils.parser import _normalize_host_key


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return round(num / den, 4) if den else default


def _pct_change(curr: float, prev: float) -> float | None:
    """Return percentage change or None when prior period is zero."""
    if prev == 0:
        return None
    return round((curr - prev) / abs(prev) * 100, 2)


def _round2(v: float) -> float:
    return round(float(v), 2)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Summary Cards
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(bookings: pd.DataFrame, host_summary: pd.DataFrame) -> dict:
    """Top-level KPI cards shown at the top of the dashboard."""
    if bookings.empty:
        return {}

    b = bookings.copy()

    total_gross   = _round2(b["Gross Sales"].sum())
    total_net     = _round2(b["Net Sales"].sum())
    total_discount = _round2(b["discount"].sum())
    total_events  = len(b)
    total_hosts   = b["host"].nunique()

    # Revenue-generating events only
    paid_mask      = b["Gross Sales"] > 0
    paid_events    = int(paid_mask.sum())
    zero_events    = total_events - paid_events

    avg_gross_per_event = _round2(_safe_div(total_gross, paid_events))
    avg_net_per_event   = _round2(_safe_div(total_net, paid_events))

    # Duration / utilisation
    total_hrs   = _round2(float(b["duration_hrs"].sum()))
    avg_hrs     = _round2(_safe_div(total_hrs, total_events))

    # Discount metrics
    overall_discount_pct = _round2(_safe_div(total_discount, total_gross) * 100)

    # Operational metrics that ONLY exist in the Host file (not dollar
    # figures — these never appear in the booking-level files at all, so
    # there is no double-counting risk in showing them here).
    total_setup   = int(host_summary["setup_count"].sum()) if not host_summary.empty else 0
    total_attend  = int(host_summary["attendance"].sum()) if not host_summary.empty else 0

    # Month range
    months_active = b["month"].nunique()
    avg_monthly_gross = _round2(_safe_div(total_gross, months_active))

    return {
        "total_gross_sales":      total_gross,
        "total_net_sales":        total_net,
        "total_discount":         total_discount,
        "overall_discount_pct":   overall_discount_pct,
        "total_events":           total_events,
        "paid_events":            paid_events,
        "zero_sales_events":      zero_events,
        "total_unique_hosts":     total_hosts,
        "avg_gross_per_event":    avg_gross_per_event,
        "avg_net_per_event":      avg_net_per_event,
        "total_room_hours":       total_hrs,
        "avg_event_duration_hrs": avg_hrs,
        "total_setup_count":      total_setup,
        "total_attendance":       total_attend,
        "months_active":          months_active,
        "avg_monthly_gross":      avg_monthly_gross,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monthly Revenue Trend
# ─────────────────────────────────────────────────────────────────────────────

def monthly_revenue_trend(bookings: pd.DataFrame) -> dict:
    """Line/bar chart: Gross Sales, Net Sales, Discount by calendar month."""
    if bookings.empty:
        return {"labels": [], "gross": [], "net": [], "discount": [], "events": []}

    grp = (
        bookings.groupby("month")
        .agg(
            gross=("Gross Sales", "sum"),
            net=("Net Sales", "sum"),
            discount=("discount", "sum"),
            events=("res_id", "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    # Human-readable label
    grp["label"] = pd.to_datetime(grp["month"]).dt.strftime("%b %Y")

    return {
        "labels":   grp["label"].tolist(),
        "gross":    [_round2(v) for v in grp["gross"]],
        "net":      [_round2(v) for v in grp["net"]],
        "discount": [_round2(v) for v in grp["discount"]],
        "events":   grp["events"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Client Segment Analysis
# ─────────────────────────────────────────────────────────────────────────────

def segment_analysis(bookings: pd.DataFrame, host_summary: pd.DataFrame) -> dict:
    """
    Pie/doughnut + bar: revenue and event count by client segment.

    Revenue ALWAYS comes from the booking-level files (Gross Sales / Net
    Sales columns on `bookings`). The Host summary file is used only
    upstream, during parsing, to assign each booking's `segment` label
    (host_type) via a name lookup — see _apply_host_type() in parser.py.
    This function never reads host_summary['gross_sales'], which avoids
    the double-counting / mismatch that occurs when the Host file's own
    totals are combined with booking-level totals (the two files are
    scoped differently — see the Data Quality tab for details).
    """
    if bookings.empty or "segment" not in bookings.columns:
        return {"labels": [], "gross": [], "pct": [], "events": []}

    grp = (
        bookings.groupby("segment")
        .agg(
            gross=("Gross Sales", "sum"),
            net=("Net Sales", "sum"),
            events=("res_id", "count"),
        )
        .reset_index()
        .sort_values("gross", ascending=False)
    )
    total = grp["gross"].sum()

    # How much of this segmentation is ground-truth (from the Host file)
    # vs the keyword fallback — surfaced so the UI/report can show it.
    source_mix = {}
    if "segment_source" in bookings.columns:
        source_mix = (
            bookings["segment_source"].value_counts(normalize=True) * 100
        ).round(1).to_dict()

    return {
        "source": "bookings",   # revenue is always booking-derived now
        "labels": grp["segment"].tolist(),
        "gross":  [_round2(v) for v in grp["gross"]],
        "net":    [_round2(v) for v in grp["net"]],
        "pct":    [_round2(_safe_div(v, total) * 100) for v in grp["gross"]],
        "events": grp["events"].tolist(),
        "segment_source_mix": source_mix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top Hosts / Clients
# ─────────────────────────────────────────────────────────────────────────────

def top_hosts(
    bookings: pd.DataFrame,
    host_summary: pd.DataFrame,
    n: int = 15,
    sort_by: str = "net",
) -> list[dict]:
    """
    Ranked table: top revenue-generating hosts.

    Revenue always comes from the booking-level Gross Sales / Net Sales
    columns. host_summary is not used here for dollar figures — the
    `segment` column on `bookings` already carries the host_type resolved
    from the Host file during parsing (see _apply_host_type in parser.py),
    so ranking and segmentation both stay consistent with the rest of the
    dashboard's revenue numbers.

    `sort_by` selects which metric ranks the top N: "gross" or "net".
    """
    if bookings.empty:
        return []

    if sort_by not in ("gross", "net"):
        sort_by = "gross"

    grp = (
        bookings.groupby("host")
        .agg(
            gross=("Gross Sales", "sum"),
            net=("Net Sales", "sum"),
            events=("res_id", "count"),
            segment=("segment", "first"),
        )
        .reset_index()
        .sort_values(sort_by, ascending=False)
        .head(n)
    )
    return [
        {
            "rank":    i + 1,
            "host":    r["host"],
            "host_type": r["segment"],
            "gross":   _round2(r["gross"]),
            "net":     _round2(r["net"]),
            "events":  int(r["events"]),
        }
        for i, r in grp.reset_index(drop=True).iterrows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Room / Space Utilisation
# ─────────────────────────────────────────────────────────────────────────────

def room_utilisation(bookings: pd.DataFrame, n: int = 20) -> dict:
    """Which rooms are booked most / generate the most revenue."""
    if bookings.empty or "room" not in bookings.columns:
        return {"labels": [], "bookings": [], "hours": [], "gross": []}

    b = bookings[bookings["room"].str.strip() != ""].copy()
    if b.empty:
        return {"labels": [], "bookings": [], "hours": [], "gross": []}

    grp = (
        b.groupby("room")
        .agg(
            bookings=("res_id", "count"),
            hours=("duration_hrs", "sum"),
            gross=("Gross Sales", "sum"),
        )
        .reset_index()
        .sort_values("bookings", ascending=False)
        .head(n)
    )

    return {
        "labels":   grp["room"].tolist(),
        "bookings": grp["bookings"].tolist(),
        "hours":    [_round2(v) for v in grp["hours"]],
        "gross":    [_round2(v) for v in grp["gross"]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Day-of-Week & Hour Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def booking_heatmap(bookings: pd.DataFrame) -> dict:
    """7×15 heatmap: number of events per weekday × start hour."""
    if bookings.empty:
        return {"days": [], "hours": [], "matrix": []}

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = list(range(6, 21))  # 6 AM – 8 PM practical range

    b = bookings.copy()
    b["weekday"] = pd.Categorical(b["weekday"], categories=days_order, ordered=True)
    b["hour_bin"] = b["hour"].clip(6, 20)

    pivot = (
        b.groupby(["weekday", "hour_bin"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=days_order, columns=hours, fill_value=0)
    )

    return {
        "days":   days_order,
        "hours":  [f"{h:02d}:00" for h in hours],
        "matrix": pivot.values.tolist(),   # list[list[int]]
    }


def weekday_summary(bookings: pd.DataFrame) -> dict:
    """Bar chart: total Gross Sales by day of week."""
    if bookings.empty:
        return {"labels": [], "total_gross": [], "total_events": []}

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grp = (
        bookings.groupby("weekday")
        .agg(total_gross=("Gross Sales", "sum"), events=("res_id", "count"))
        .reindex(days_order)
        .fillna(0)
        .reset_index()
    )
    return {
        "labels":       grp["weekday"].tolist(),
        "total_gross":  [_round2(v) for v in grp["total_gross"]],
        "total_events": grp["events"].astype(int).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Event Status Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def status_breakdown(bookings: pd.DataFrame) -> dict:
    """Pie chart: Reserved / Tentative / Cancelled etc."""
    if bookings.empty:
        return {"labels": [], "counts": [], "gross": []}

    # Normalise status labels
    b = bookings.copy()
    b["status_clean"] = (
        b["status"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.title()
    )
    b["status_clean"] = b["status_clean"].replace(
        {"Tentative Reservation": "Tentative", "": "Unknown"}
    )

    grp = (
        b.groupby("status_clean")
        .agg(count=("res_id", "count"), gross=("Gross Sales", "sum"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return {
        "labels": grp["status_clean"].tolist(),
        "counts": grp["count"].tolist(),
        "gross":  [_round2(v) for v in grp["gross"]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Payment Type Analysis
# ─────────────────────────────────────────────────────────────────────────────

def payment_type_analysis(bookings: pd.DataFrame) -> dict:
    """Bar: revenue by payment/billing type."""
    if bookings.empty:
        return {"labels": [], "gross": [], "net": [], "events": []}

    b = bookings.copy()
    b["pt"] = (
        b["payment_type"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"(none)": "No Payment Type", "": "No Payment Type"})
    )

    grp = (
        b.groupby("pt")
        .agg(gross=("Gross Sales", "sum"), net=("Net Sales", "sum"), events=("res_id", "count"))
        .reset_index()
        .sort_values("gross", ascending=False)
    )
    return {
        "labels": grp["pt"].tolist(),
        "gross":  [_round2(v) for v in grp["gross"]],
        "net":    [_round2(v) for v in grp["net"]],
        "events": grp["events"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Discount Analysis
# ─────────────────────────────────────────────────────────────────────────────

def discount_analysis(bookings: pd.DataFrame) -> dict:
    """Scatter / bar: who gets discounts and how much."""
    if bookings.empty:
        return {"by_segment": {}, "top_discounted": []}

    b = bookings[bookings["discount"] > 0].copy()

    by_seg = (
        b.groupby("segment")
        .agg(
            total_discount=("discount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            events=("res_id", "count"),
        )
        .reset_index()
        .sort_values("total_discount", ascending=False)
    )

    top_disc = (
        b.groupby("host")
        .agg(
            gross=("Gross Sales", "sum"),
            net=("Net Sales", "sum"),
            discount=("discount", "sum"),
            avg_pct=("discount_pct", "mean"),
        )
        .reset_index()
        .sort_values("discount", ascending=False)
        .head(10)
    )

    return {
        "by_segment": {
            "labels":           by_seg["segment"].tolist(),
            "total_discount":   [_round2(v) for v in by_seg["total_discount"]],
            "avg_discount_pct": [_round2(v) for v in by_seg["avg_discount_pct"]],
        },
        "top_discounted": [
            {
                "host":     r["host"],
                "gross":    _round2(r["gross"]),
                "net":      _round2(r["net"]),
                "discount": _round2(r["discount"]),
                "avg_pct":  _round2(r["avg_pct"]),
            }
            for _, r in top_disc.iterrows()
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly Summary
# ─────────────────────────────────────────────────────────────────────────────

def quarterly_summary(bookings: pd.DataFrame) -> dict:
    """Grouped bar chart: Q1–Q4 gross vs net (fiscal year aware)."""
    if bookings.empty:
        return {"labels": [], "gross": [], "net": [], "events": []}

    b = bookings.copy()
    # Fiscal quarter: FY starts July → Jul/Aug/Sep = FQ1
    b["fiscal_month"] = b["start"].dt.month
    b["fq"] = b["fiscal_month"].apply(
        lambda m: f"FQ{((m - 7) % 12) // 3 + 1}"
    )
    b["fy_fq"] = b["fiscal_year"] + " " + b["fq"]

    grp = (
        b.groupby("fy_fq")
        .agg(gross=("Gross Sales", "sum"), net=("Net Sales", "sum"), events=("res_id", "count"))
        .reset_index()
        .sort_values("fy_fq")
    )
    return {
        "labels": grp["fy_fq"].tolist(),
        "gross":  [_round2(v) for v in grp["gross"]],
        "net":    [_round2(v) for v in grp["net"]],
        "events": grp["events"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Event Duration Distribution
# ─────────────────────────────────────────────────────────────────────────────

def duration_distribution(bookings: pd.DataFrame) -> dict:
    """Histogram buckets for event length in hours."""
    if bookings.empty:
        return {"labels": [], "counts": []}

    bins   = [0, 1, 2, 4, 6, 8, 10, 12, 24, 48, 999]
    labels = ["<1h", "1-2h", "2-4h", "4-6h", "6-8h", "8-10h", "10-12h", "12-24h", "24-48h", "48h+"]

    hours = bookings["duration_hrs"].clip(upper=999)
    counts, _ = np.histogram(hours, bins=bins)

    return {"labels": labels, "counts": counts.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Revenue by Host Type (from host_summary detail)
# ─────────────────────────────────────────────────────────────────────────────

def host_type_detail(bookings: pd.DataFrame, host_summary: pd.DataFrame) -> dict:
    """
    Full ranked list within each host_type/segment for the drilldown table
    and the Host Type Breakdown chart.

    Revenue (gross_sales, total) is always computed from the booking-level
    Gross Sales column. host_summary is used only to look up an
    informational setup_count per host (a count that exists ONLY in the
    Host file, never a dollar figure), so it can never cause a revenue
    mismatch with the rest of the dashboard.
    """
    if bookings.empty or "segment" not in bookings.columns:
        return {}

    # Optional: host -> setup_count lookup, purely informational.
    setup_lookup = {}
    if not host_summary.empty:
        for _, row in host_summary.iterrows():
            key = _normalize_host_key(row["host"])
            setup_lookup[key] = setup_lookup.get(key, 0) + int(row["setup_count"])

    out = {}
    for ht, grp in bookings.groupby("segment"):
        host_grp = (
            grp.groupby("host")
            .agg(gross_sales=("Gross Sales", "sum"))
            .reset_index()
            .sort_values("gross_sales", ascending=False)
        )
        host_grp["setup_count"] = host_grp["host"].apply(
            lambda h: setup_lookup.get(_normalize_host_key(h), 0)
        )
        out[ht] = {
            "hosts":       host_grp["host"].tolist(),
            "gross_sales": [_round2(v) for v in host_grp["gross_sales"]],
            "setup_count": host_grp["setup_count"].tolist(),
            "total":       _round2(host_grp["gross_sales"].sum()),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Monthly Booking Volume (events count trend)
# ─────────────────────────────────────────────────────────────────────────────

def monthly_event_volume(bookings: pd.DataFrame) -> dict:
    if bookings.empty:
        return {"labels": [], "events": [], "paid_events": []}

    b = bookings.copy()
    grp = b.groupby("month").agg(
        events=("res_id", "count"),
        paid=("Gross Sales", lambda x: (x > 0).sum()),
    ).reset_index().sort_values("month")
    grp["label"] = pd.to_datetime(grp["month"]).dt.strftime("%b %Y")

    return {
        "labels":      grp["label"].tolist(),
        "events":      grp["events"].tolist(),
        "paid_events": grp["paid"].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full dashboard payload
# ─────────────────────────────────────────────────────────────────────────────

def build_full_dashboard(
    bookings: pd.DataFrame,
    host_summary: pd.DataFrame,
    reporting_period: str,
    validation: Any,
) -> dict:
    """Assemble every chart and KPI into one JSON-serialisable dict."""
    return {
        "reporting_period":    reporting_period,
        "kpis":                compute_kpis(bookings, host_summary),
        "monthly_revenue":     monthly_revenue_trend(bookings),
        "monthly_volume":      monthly_event_volume(bookings),
        "segment_analysis":    segment_analysis(bookings, host_summary),
        "top_hosts":           top_hosts(bookings, host_summary),
        "room_utilisation":    room_utilisation(bookings),
        "booking_heatmap":     booking_heatmap(bookings),
        "weekday_summary":     weekday_summary(bookings),
        "status_breakdown":    status_breakdown(bookings),
        "payment_analysis":    payment_type_analysis(bookings),
        "discount_analysis":   discount_analysis(bookings),
        "quarterly_summary":   quarterly_summary(bookings),
        "duration_dist":       duration_distribution(bookings),
        "host_type_detail":    host_type_detail(bookings, host_summary),
        "validation":          validation.to_dict() if hasattr(validation, "to_dict") else validation,
    }
