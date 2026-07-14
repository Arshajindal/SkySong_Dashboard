"""
API Blueprint – JSON endpoints consumed by the dashboard frontend.
All endpoints require data to be loaded first.
"""
import math
from flask import Blueprint, jsonify, request, current_app
from app.models.store import is_loaded, get_store
from app.utils.analytics import (
    build_full_dashboard,
    compute_kpis,
    monthly_revenue_trend,
    monthly_event_volume,
    segment_analysis,
    top_hosts,
    room_utilisation,
    booking_heatmap,
    weekday_summary,
    status_breakdown,
    payment_type_analysis,
    discount_analysis,
    quarterly_summary,
    duration_distribution,
    host_type_detail,
)

api_bp = Blueprint("api", __name__)


def _require_data():
    if not is_loaded():
        return jsonify({"error": "No data loaded. Please upload files first."}), 404
    return None


def _clean(obj):
    """Recursively replace NaN/Inf so jsonify doesn't choke."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Full dashboard payload (single call to hydrate the whole page)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/dashboard")
def api_dashboard():
    err = _require_data()
    if err:
        return err
    store = get_store()
    payload = build_full_dashboard(
        store.bookings,
        store.host_summary,
        store.reporting_period,
        store.validation,
    )
    return jsonify(_clean(payload))


# ─────────────────────────────────────────────────────────────────────────────
# Individual chart endpoints (for lazy-loading / tab switching)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/kpis")
def api_kpis():
    err = _require_data()
    if err: return err
    s = get_store()
    return jsonify(_clean(compute_kpis(s.bookings, s.host_summary)))


@api_bp.route("/monthly-revenue")
def api_monthly_revenue():
    err = _require_data()
    if err: return err
    return jsonify(_clean(monthly_revenue_trend(get_store().bookings)))


@api_bp.route("/monthly-volume")
def api_monthly_volume():
    err = _require_data()
    if err: return err
    return jsonify(_clean(monthly_event_volume(get_store().bookings)))


@api_bp.route("/segments")
def api_segments():
    err = _require_data()
    if err: return err
    s = get_store()
    return jsonify(_clean(segment_analysis(s.bookings, s.host_summary)))


@api_bp.route("/top-hosts")
def api_top_hosts():
    err = _require_data()
    if err: return err
    n = request.args.get("n", 15, type=int)
    sort_by = request.args.get("sort_by", "net", type=str)
    s = get_store()
    return jsonify(_clean(top_hosts(s.bookings, s.host_summary, n=n, sort_by=sort_by)))


@api_bp.route("/rooms")
def api_rooms():
    err = _require_data()
    if err: return err
    return jsonify(_clean(room_utilisation(get_store().bookings)))


@api_bp.route("/heatmap")
def api_heatmap():
    err = _require_data()
    if err: return err
    return jsonify(_clean(booking_heatmap(get_store().bookings)))


@api_bp.route("/weekday")
def api_weekday():
    err = _require_data()
    if err: return err
    return jsonify(_clean(weekday_summary(get_store().bookings)))


@api_bp.route("/status")
def api_status():
    err = _require_data()
    if err: return err
    return jsonify(_clean(status_breakdown(get_store().bookings)))


@api_bp.route("/payment-types")
def api_payment_types():
    err = _require_data()
    if err: return err
    return jsonify(_clean(payment_type_analysis(get_store().bookings)))


@api_bp.route("/discounts")
def api_discounts():
    err = _require_data()
    if err: return err
    return jsonify(_clean(discount_analysis(get_store().bookings)))


@api_bp.route("/quarterly")
def api_quarterly():
    err = _require_data()
    if err: return err
    return jsonify(_clean(quarterly_summary(get_store().bookings)))


@api_bp.route("/duration")
def api_duration():
    err = _require_data()
    if err: return err
    return jsonify(_clean(duration_distribution(get_store().bookings)))


@api_bp.route("/host-types")
def api_host_types():
    err = _require_data()
    if err: return err
    s = get_store()
    return jsonify(_clean(host_type_detail(s.bookings, s.host_summary)))


# ─────────────────────────────────────────────────────────────────────────────
# Bookings table (paginated)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/bookings")
def api_bookings():
    err = _require_data()
    if err: return err
    s   = get_store()
    b   = s.bookings

    page     = request.args.get("page",     1,    type=int)
    per_page = request.args.get("per_page", 50,   type=int)
    search   = request.args.get("q",        "",   type=str).lower()
    segment  = request.args.get("segment",  "",   type=str)
    status   = request.args.get("status",   "",   type=str)
    sort_by  = request.args.get("sort",     "start")
    order    = request.args.get("order",    "asc")

    df = b.copy()

    # Filters
    if search:
        mask = (
            df["host"].str.lower().str.contains(search, na=False) |
            df["event_name"].str.lower().str.contains(search, na=False) |
            df["room"].str.lower().str.contains(search, na=False)
        )
        df = df[mask]

    if segment:
        df = df[df["segment"].str.lower() == segment.lower()]

    if status:
        df = df[df["status"].str.lower().str.contains(status.lower(), na=False)]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(order == "asc"))

    total = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    # Serialise datetimes
    cols = ["start", "end", "host", "event_name", "room", "status",
            "payment_type", "res_id", "Gross Sales", "Net Sales",
            "discount", "discount_pct", "segment", "duration_hrs"]
    cols = [c for c in cols if c in page_df.columns]
    rows = page_df[cols].copy()
    rows["start"] = rows["start"].dt.strftime("%Y-%m-%d %H:%M")
    rows["end"]   = rows["end"].dt.strftime("%Y-%m-%d %H:%M")

    return jsonify({
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "pages":      math.ceil(total / per_page),
        "rows":       _clean(rows.to_dict(orient="records")),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Health / meta
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/health")
def health():
    s = get_store()
    return jsonify({
        "loaded":           is_loaded(),
        "rows":             len(s.bookings) if is_loaded() else 0,
        "reporting_period": s.reporting_period,
        "source_files":     s.source_files,
    })
