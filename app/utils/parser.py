"""
EMS Data Parser & Cleaner
=========================
Parses the three EMS report formats exported from the booking system:
  1. Net Sales by Booking   – row-per-event with merged cells
  2. Gross Sales by Booking – same layout, different money column
  3. Gross Sales by Host    – summary grouped by host type / host name

Design decisions
----------------
- Column positions are NEVER hardcoded. Every field (Start, End, Host,
  Res ID, Net Sales, Gross Sales, ...) is located at runtime by scanning
  the report's header block for label text ("dynamic schema resolution",
  see `_build_field_map`). If a future export reorders, adds, or renames
  columns, the parser adapts instead of silently reading the wrong field.
  Callers may also supply an explicit `schema_map` override for files
  whose headers don't match the built-in aliases (see `parse_ems_files`).
- The Net/Gross "role" of a booking file (which money column it holds)
  is auto-detected from its header, not assumed from the caller's label.
- We NEVER impute or fabricate missing monetary values.
  If a row genuinely has no sales figure we record 0.0 (the EMS system
  exports a blank cell when a booking is $0, e.g. internal-ASU events).
- Data-quality issues (duplicate Res IDs, mismatched Net vs Gross,
  non-numeric sales, merge reconciliation failures) are logged in a
  validation report returned alongside the parsed frames so the UI can
  surface them.
- All string cleaning is limited to whitespace/newline normalisation and
  capitalisation; we do not rename hosts or merge similar-looking names
  automatically (that would silently alter business data).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SchemaOverride = dict  # {canonical_field: column_index (int) | header text (str)}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    total_rows_raw: int = 0
    total_rows_parsed: int = 0
    rows_skipped_header: int = 0
    rows_skipped_total: int = 0
    rows_skipped_bad_date: int = 0
    zero_sales_count: int = 0
    duplicate_res_ids: list = field(default_factory=list)
    mismatched_net_gross: list = field(default_factory=list)  # res_ids where values differ by >1%
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self):
        return {
            "total_rows_raw": self.total_rows_raw,
            "total_rows_parsed": self.total_rows_parsed,
            "rows_skipped_header": self.rows_skipped_header,
            "rows_skipped_total": self.rows_skipped_total,
            "rows_skipped_bad_date": self.rows_skipped_bad_date,
            "zero_sales_count": self.zero_sales_count,
            "duplicate_res_ids_count": len(self.duplicate_res_ids),
            "mismatch_count": len(self.mismatched_net_gross),
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ParsedDataset:
    bookings: pd.DataFrame = field(default_factory=pd.DataFrame)
    host_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: ValidationReport = field(default_factory=ValidationReport)
    reporting_period: str = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_str(v) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v).replace("\n", " ")).strip()


def _safe_float(v) -> Optional[float]:
    """Return float or None – never fabricate."""
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _is_total_row(row_series) -> bool:
    """Detect subtotal / grand-total / page-footer rows."""
    for cell in row_series:
        s = _clean_str(cell)
        if s in ("Date Total", "Month Total", "Grand Total"):
            return True
        if re.match(r"^Page \d+ of \d+$", s):
            return True
    return False


def _extract_reporting_period(raw_df: pd.DataFrame) -> str:
    """Pull the 'Reporting Period: ...' text from the header block."""
    for _, row in raw_df.iterrows():
        for cell in row:
            s = _clean_str(cell)
            if s.startswith("Reporting Period:"):
                return s.replace("Reporting Period:", "").strip()
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic schema resolution
# ─────────────────────────────────────────────────────────────────────────────
#
# Every EMS export wraps its column headers across a few stacked rows above
# the first data row (e.g. "Payment Type" and "Res ID" appear several rows
# apart from "Start"/"End"/"Host"). Rather than hardcode column positions
# for one specific layout, we scan the header block for label text and
# build a {canonical_field: column_index} map at parse time. This is what
# lets the same code handle a file whose columns have shifted, been added
# to, or been reordered — the requirement is "derive schema/headers/values
# dynamically", not "assume today's export forever".

BOOKING_FIELD_ALIASES: dict[str, list[str]] = {
    "start":        ["start"],
    "end":          ["end"],
    "host":         ["host"],
    "event_name":   ["event name/location", "event name", "location"],
    "payment_type": ["payment type", "billing type"],
    "status":       ["booking status", "status"],
    "res_id":       ["res id", "reservation id"],
    "net_sales":    ["net sales", "net revenue", "net amount"],
    "gross_sales":  ["gross sales", "gross revenue", "gross amount"],
}
BOOKING_REQUIRED_FIELDS = ("start", "end", "host", "res_id")

HOST_FIELD_ALIASES: dict[str, list[str]] = {
    "host_type":   ["host type", "client type", "segment"],
    "host":        ["host", "client"],
    "setup_count": ["setup count", "setups"],
    "attendance":  ["attendance", "attendees"],
    "gross_sales": ["gross sales", "gross revenue"],
}
HOST_REQUIRED_FIELDS = ("host", "gross_sales")


def _build_field_map(
    raw_df: pd.DataFrame,
    field_aliases: dict[str, list[str]],
    header_rows: int = 12,
) -> dict[str, int]:
    """
    Discover which column holds each canonical field by scanning the
    report's header block (the first `header_rows` rows) for label text,
    accumulating matches across the whole block since EMS headers are
    frequently split across several stacked rows rather than one.

    Aliases are matched longest-first so a specific label (e.g. "host
    type") claims its column before a more generic one (e.g. "host") is
    considered for a different field, and each column is claimed at most
    once.
    """
    n_rows = min(header_rows, len(raw_df))
    n_cols = raw_df.shape[1]
    cells: list[tuple[int, str]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            text = _clean_str(raw_df.iat[r, c]).lower()
            if text:
                cells.append((c, text))

    alias_items = sorted(
        ((f, a) for f, aliases in field_aliases.items() for a in aliases),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    # Two passes: exact-text matches first, then substring fallback. This
    # matters because report titles/captions (e.g. "Sales by Host") can
    # contain a field's alias as a substring ("host") — an exact-match pass
    # finds the real "Host" header column first and locks it in before the
    # substring pass would otherwise mis-claim the title cell.
    field_map: dict[str, int] = {}
    used_cols: set[int] = set()
    for match_mode in ("exact", "contains"):
        for f, alias in alias_items:
            if f in field_map:
                continue
            for col, text in cells:
                if col in used_cols:
                    continue
                hit = (text == alias) if match_mode == "exact" else (alias in text)
                if hit:
                    field_map[f] = col
                    used_cols.add(col)
                    break
    return field_map


def _apply_schema_overrides(
    raw_df: pd.DataFrame,
    field_map: dict[str, int],
    schema_map: Optional[SchemaOverride],
) -> dict[str, int]:
    """
    Layer caller-supplied overrides on top of the auto-detected field map.
    Each override may be an explicit column index (int) or header text to
    search for (str) — the manual escape hatch for files whose columns
    differ enough that keyword auto-detection can't resolve them.
    """
    if not schema_map:
        return field_map
    resolved = dict(field_map)
    for f, override in schema_map.items():
        if isinstance(override, bool):
            continue
        if isinstance(override, int):
            resolved[f] = override
        elif isinstance(override, str):
            match = _build_field_map(raw_df, {f: [override.lower()]})
            if f in match:
                resolved[f] = match[f]
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Booking file parser  (Net Sales & Gross Sales share the same layout)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_booking_sheet(
    raw_df: pd.DataFrame,
    report: ValidationReport,
    schema_map: Optional[SchemaOverride] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Parse a "Sales by Booking" export (Net or Gross flavour). Every field's
    column position — including whether this file holds Net Sales or Gross
    Sales — is resolved at runtime from the header block via
    `_build_field_map`; nothing is assumed from a fixed layout or from the
    caller's file-role label.

    The EMS export is a merged-cell report:
      • Row pattern per booking:
          Row A: start, end, host, event_name, payment_type, status, res_id, sales
          Row B: blank except the Res ID column, which holds the Book ID
          Row C: blank except the Event Name column, which holds the Room name
        (sometimes rows B and C are swapped or the room appears on a 4th row)
      • Date-header rows: start = date@midnight, end = NaT
      • Total rows: 'Date Total' / 'Month Total' / 'Grand Total'
      • Page break rows: 'Page N of N'

    Returns (dataframe, sales_col_name) where sales_col_name is
    "Net Sales" or "Gross Sales", whichever was found in this file's header.
    """
    report.total_rows_raw += len(raw_df)

    field_map = _build_field_map(raw_df, BOOKING_FIELD_ALIASES)
    field_map = _apply_schema_overrides(raw_df, field_map, schema_map)

    missing = [f for f in BOOKING_REQUIRED_FIELDS if f not in field_map]
    if missing:
        report.errors.append(
            f"Could not locate required column(s) {missing} in the booking file "
            "header — parsing aborted. Pass a schema_map override if this file "
            "uses non-standard column labels."
        )
        return pd.DataFrame(), "Sales"

    has_net, has_gross = "net_sales" in field_map, "gross_sales" in field_map
    if has_net and has_gross:
        report.warnings.append(
            "Both 'Net Sales' and 'Gross Sales' headers were found in the same "
            "booking file; defaulting to Net Sales. Pass schema_map to disambiguate."
        )
        sales_field, sales_col_name = "net_sales", "Net Sales"
    elif has_net:
        sales_field, sales_col_name = "net_sales", "Net Sales"
    elif has_gross:
        sales_field, sales_col_name = "gross_sales", "Gross Sales"
    else:
        report.errors.append(
            "Could not locate a 'Net Sales' or 'Gross Sales' column in the "
            "booking file header — parsing aborted."
        )
        return pd.DataFrame(), "Sales"

    col_start  = field_map["start"]
    col_end    = field_map["end"]
    col_host   = field_map["host"]
    col_event  = field_map.get("event_name")      # also doubles as the room column
    col_pay    = field_map.get("payment_type")
    col_status = field_map.get("status")
    col_resid  = field_map["res_id"]               # also doubles as the book_id column
    col_sales  = field_map[sales_field]

    records = []
    pending: dict = {}

    for _, row in raw_df.iterrows():
        # ── Skip total / page-footer rows ─────────────────────────────────────
        if _is_total_row(row):
            report.rows_skipped_total += 1
            if pending:
                records.append(pending)
                pending = {}
            continue

        v_start  = row.iloc[col_start]
        v_end    = row.iloc[col_end]
        v_host   = row.iloc[col_host]
        v_event  = row.iloc[col_event]  if col_event  is not None else np.nan
        v_pay    = row.iloc[col_pay]    if col_pay    is not None else np.nan
        v_status = row.iloc[col_status] if col_status is not None else np.nan
        v_resid  = row.iloc[col_resid]
        v_sales  = row.iloc[col_sales]

        # ── Date-header row (midnight, no end time, no host) ──────────────────
        if pd.notna(v_start) and pd.isna(v_end) and pd.isna(v_host):
            report.rows_skipped_header += 1
            continue

        # ── Event data row (has start, end, and host) ─────────────────────────
        if pd.notna(v_start) and pd.notna(v_end) and pd.notna(v_host):
            if pending:
                records.append(pending)
                pending = {}

            try:
                start = pd.to_datetime(v_start)
                end   = pd.to_datetime(v_end)
            except Exception:
                report.rows_skipped_bad_date += 1
                continue

            sales_raw = _safe_float(v_sales)
            sales = 0.0 if sales_raw is None else sales_raw
            if sales_raw is None and _clean_str(v_sales) not in ("", "nan"):
                report.warnings.append(
                    f"Non-numeric {sales_col_name} '{v_sales}' on "
                    f"{start.date()} – {_clean_str(v_host)[:40]}; treated as 0."
                )

            pending = {
                "start":        start,
                "end":          end,
                "duration_hrs": round((end - start).total_seconds() / 3600, 2),
                "host":         _clean_str(v_host),
                "event_name":   _clean_str(v_event),
                "payment_type": _clean_str(v_pay),
                "status":       _clean_str(v_status),
                "res_id":       _clean_str(v_resid) if pd.notna(v_resid) else "",
                "book_id":      "",
                "room":         "",
                sales_col_name: sales,
                "month":        start.strftime("%Y-%m"),
                "month_label":  start.strftime("%b %Y"),
                "weekday":      start.strftime("%A"),
                "hour":         start.hour,
                "date":         start.date(),
                "year":         start.year,
                "quarter":      f"Q{start.quarter}",
            }
            continue

        # ── Continuation row (book_id or room) ────────────────────────────────
        if pending:
            # Book ID row: res_id column has a numeric-ish string, event col is blank
            if pd.notna(v_resid) and pd.isna(v_event):
                pending["book_id"] = _clean_str(v_resid)
            # Room row: event_name column has the room name
            if pd.notna(v_event) and pd.isna(v_start):
                existing_room = pending.get("room", "")
                room_str = _clean_str(v_event)
                if existing_room == "":
                    pending["room"] = room_str
                # else already set; EMS sometimes repeats room on a 3rd continuation row

    # Flush last pending
    if pending:
        records.append(pending)

    report.total_rows_parsed += len(records)
    df = pd.DataFrame(records)

    if df.empty:
        return df, sales_col_name

    # ── Post-parse cleanup ────────────────────────────────────────────────────
    df[sales_col_name] = pd.to_numeric(df[sales_col_name], errors="coerce").fillna(0.0)
    df["start"] = pd.to_datetime(df["start"])
    df["end"]   = pd.to_datetime(df["end"])

    # Count genuine zeros (internal/ASU events billed at $0)
    report.zero_sales_count = int((df[sales_col_name] == 0).sum())

    return df, sales_col_name


# ─────────────────────────────────────────────────────────────────────────────
# Host summary file parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_host_sheet(
    raw_df: pd.DataFrame,
    schema_map: Optional[SchemaOverride] = None,
) -> pd.DataFrame:
    """
    Parse the Host summary report. Column positions are resolved at
    runtime from the header block (Host Type / Host / Setup Count /
    Attendance / Gross Sales), not assumed from a fixed layout.
    """
    field_map = _build_field_map(raw_df, HOST_FIELD_ALIASES)
    field_map = _apply_schema_overrides(raw_df, field_map, schema_map)

    missing = [f for f in HOST_REQUIRED_FIELDS if f not in field_map]
    if missing:
        return pd.DataFrame()

    col_host_type = field_map.get("host_type")
    col_host      = field_map["host"]
    col_setup     = field_map.get("setup_count")
    col_attend    = field_map.get("attendance")
    col_gross     = field_map["gross_sales"]

    records = []
    current_host_type = "Unknown"

    for _, row in raw_df.iterrows():
        v_type   = row.iloc[col_host_type] if col_host_type is not None else np.nan
        v_host   = row.iloc[col_host]
        v_setup  = row.iloc[col_setup]  if col_setup  is not None else np.nan
        v_attend = row.iloc[col_attend] if col_attend is not None else np.nan
        v_gross  = row.iloc[col_gross]

        # Update host type header
        s_type = _clean_str(v_type)
        if s_type and s_type not in ("Host Type", "Total", "Grand Total"):
            current_host_type = s_type

        # Skip non-data rows
        if pd.isna(v_host):
            continue
        host_name = _clean_str(v_host)
        if not host_name or host_name in ("Host", "nan"):
            continue
        if re.match(r"^Page \d+ of \d+$", host_name):
            continue

        setup  = _safe_float(v_setup)
        attend = _safe_float(v_attend)
        gross  = _safe_float(v_gross)

        # Skip total aggregation rows (setup column may say "Total")
        if _clean_str(v_setup) == "Total":
            continue

        if gross is None:
            continue   # row has no monetary data – skip silently

        records.append({
            "host_type":   current_host_type,
            "host":        host_name,
            "setup_count": int(setup)  if setup  is not None else 0,
            "attendance":  int(attend) if attend is not None else 0,
            "gross_sales": gross,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["gross_sales"] = pd.to_numeric(df["gross_sales"], errors="coerce").fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-file validation
# ─────────────────────────────────────────────────────────────────────────────

def _cross_validate(df_net: pd.DataFrame, df_gross: pd.DataFrame, report: ValidationReport):
    """
    Check that where res_id matches between the two booking files,
    gross >= net (discounts reduce net from gross; they cannot be inverted).
    We flag mismatches but do NOT alter values.
    """
    if df_net.empty or df_gross.empty:
        return

    if "res_id" not in df_net.columns or "res_id" not in df_gross.columns:
        return
    if "Net Sales" not in df_net.columns or "Gross Sales" not in df_gross.columns:
        return

    net_by_res  = df_net.groupby("res_id")["Net Sales"].sum()
    gross_by_res = df_gross.groupby("res_id")["Gross Sales"].sum()

    common = net_by_res.index.intersection(gross_by_res.index)
    for rid in common:
        n = net_by_res[rid]
        g = gross_by_res[rid]
        if g > 0 and n > g * 1.01:   # allow 1 % float tolerance
            report.mismatched_net_gross.append(
                {"res_id": rid, "net": round(n, 2), "gross": round(g, 2)}
            )
            report.warnings.append(
                f"Res {rid}: Net Sales ({n:.2f}) > Gross Sales ({g:.2f}). "
                "Values retained as-is; please verify in EMS."
            )

    # Duplicate res_ids within the same file
    dupes = df_net[df_net["res_id"].duplicated(keep=False)]["res_id"].unique().tolist()
    if dupes:
        report.duplicate_res_ids = dupes[:20]   # cap list length
        report.warnings.append(
            f"{len(dupes)} Res IDs appear more than once in the Net Sales file "
            "(multi-room bookings or repeated date blocks – expected behaviour)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Classify host segment
# ─────────────────────────────────────────────────────────────────────────────

# Hosts that must always resolve to their own standalone segment, overriding
# whatever host_type the Host summary file assigns them. ASU EdPlus is its
# own peer entity, not a SkySong tenant, regardless of how the source Excel
# groups it.
SEGMENT_OVERRIDES: dict[str, str] = {
    "ASU EDPLUS": "ASU EdPlus",
}


def _classify_host_segment(host: str) -> str:
    """
    Derive a client-segment label from the host name using keyword matching.
    This is the FALLBACK method, used only when a host cannot be matched
    against the Host summary file's explicit host_type column (see
    _build_host_type_lookup below, which is the preferred source of truth).
    """
    h = host.upper()
    if h.startswith("ASU ") or h.startswith("ARIZONA STATE"):
        return "ASU"
    if any(kw in h for kw in ("SKYSONG", "SKY SONG")):
        return "SkySong"
    if any(kw in h for kw in ("GOVERNMENT", "COUNTY", "STATE ", "FEDERAL", "CITY OF", "DEPARTMENT")):
        return "Government"
    if any(kw in h for kw in ("SCHOOL", "ACADEMY", "UNIVERSITY", "COLLEGE", "EDUCATION", "DISTRICT")):
        return "Education"
    if any(kw in h for kw in ("HEALTH", "MEDICAL", "HOSPITAL", "CLINIC", "NURSING")):
        return "Healthcare"
    if any(kw in h for kw in ("TECHNOLOGY", "SOFTWARE", "DIGITAL", "CYBER", "COMPUTING", "CLOUD", "AI ", "DATA ")):
        return "Technology"
    if any(kw in h for kw in ("ASSOCIATION", "SOCIETY", "ALLIANCE", "INSTITUTE", "FOUNDATION", "NONPROFIT", "NON-PROFIT")):
        return "Non-Profit / Association"
    return "Commercial / Other"


def _normalize_host_key(name: str) -> str:
    """
    Normalise a host name into a matching key so that harmless formatting
    differences between the two files (extra whitespace introduced when
    Excel's wrapped-text newlines collapse next to a hyphen, e.g.
    'Well- Being' vs 'Well-Being') don't prevent a legitimate match.
    This key is used ONLY for matching; the original display name from
    the booking file is always preserved unchanged in the output.
    """
    key = re.sub(r"\s+", " ", name).strip().upper()
    key = re.sub(r"\s*-\s*", "-", key)   # normalise spacing around hyphens
    return key


def _build_host_type_lookup(host_summary: pd.DataFrame) -> dict:
    """
    Build a {normalised_host_name: host_type} lookup from the Host summary
    file. This is the ONLY thing the Host file is used for in downstream
    analytics — it is a client/host relationship (name -> type) lookup,
    never a source of dollar figures. Revenue always comes from the
    booking-level Net/Gross files.
    """
    if host_summary.empty:
        return {}
    lookup = {}
    for _, row in host_summary.iterrows():
        key = _normalize_host_key(row["host"])
        # If the same normalised host appears twice with different types
        # (shouldn't happen, but data can surprise you), keep the first
        # and don't silently overwrite — this would be a data-quality flag.
        lookup.setdefault(key, row["host_type"])
    return lookup


def _apply_host_type(bookings: pd.DataFrame, host_summary: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """
    Attach a 'segment' column to bookings using the Host file's host_type
    as ground truth wherever a host name can be matched, falling back to
    the keyword heuristic only for hosts that don't appear in the Host
    file. Also records match-rate statistics in the validation report so
    the user can see how much of the segmentation is ground-truth vs
    heuristic.
    """
    lookup = _build_host_type_lookup(host_summary)

    def _resolve(host: str) -> tuple[str, str]:
        key = _normalize_host_key(host)
        if key in SEGMENT_OVERRIDES:
            return SEGMENT_OVERRIDES[key], "override"
        if key in lookup:
            return lookup[key], "host_file"
        return _classify_host_segment(host), "heuristic"

    resolved = bookings["host"].apply(_resolve)
    bookings = bookings.copy()
    bookings["segment"] = resolved.apply(lambda t: t[0])
    bookings["segment_source"] = resolved.apply(lambda t: t[1])

    if lookup:
        matched = int((bookings["segment_source"] == "host_file").sum())
        total = len(bookings)
        report.warnings.append(
            f"Client segmentation: {matched}/{total} bookings ({matched/total*100:.1f}%) "
            f"matched to a host_type from the Host summary file; the remainder used "
            f"keyword-based classification as a fallback."
        )

    return bookings


# ─────────────────────────────────────────────────────────────────────────────
# Net/Gross merge (with reconciliation guard)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_net_and_gross(
    df_net: pd.DataFrame,
    df_gross: pd.DataFrame,
    report: ValidationReport,
) -> pd.DataFrame:
    """
    Combine the Net and Gross booking frames into one row-per-booking table.

    IMPORTANT: (res_id, start) is NOT a unique key on its own — EMS assigns
    the same reservation ID to recurring / multi-room bookings that share a
    start timestamp. Joining on that pair directly produces a many-to-many
    merge: every duplicate-key row on the left is cross-joined against every
    duplicate-key row on the right sharing that key, multiplying (not just
    duplicating) their dollar totals. That was the root cause of the
    dashboard reporting Gross/Net totals roughly 1.5x too high.

    Fix: disambiguate duplicate keys with a per-group occurrence index
    (`cumcount`) before merging, so the join is always 1:1. Both files are
    exports of the same underlying booking sequence, so the Nth occurrence
    of a given (res_id, start) in the Net file corresponds to the Nth
    occurrence in the Gross file.

    A reconciliation check re-sums both files independently and compares
    against the merged totals; any drift is logged as a hard error instead
    of being silently returned, so this class of bug fails loudly next time.
    """
    if df_net.empty or df_gross.empty:
        return df_net if not df_net.empty else df_gross

    merge_keys = ["res_id", "start"]
    df_net = df_net.copy()
    df_gross = df_gross.copy()
    df_net["_occurrence"] = df_net.groupby(merge_keys).cumcount()
    df_gross["_occurrence"] = df_gross.groupby(merge_keys).cumcount()
    join_keys = merge_keys + ["_occurrence"]

    gross_cols = join_keys + ["Gross Sales"]
    bookings = df_net.merge(
        df_gross[gross_cols],
        on=join_keys,
        how="left",
        suffixes=("", "_gross"),
    )
    bookings.drop(columns="_occurrence", inplace=True)
    df_net.drop(columns="_occurrence", inplace=True)
    # Fill unmatched gross with 0 (do NOT fabricate)
    bookings["Gross Sales"] = bookings["Gross Sales"].fillna(0.0)

    dup_groups = int((df_net.groupby(merge_keys).size() > 1).sum())
    if dup_groups:
        report.warnings.append(
            f"{dup_groups} (Res ID, Start) combinations repeat within the Net "
            "Sales file (multi-room / recurring bookings sharing a reservation "
            "ID); disambiguated by occurrence order before merging with Gross."
        )

    # ── Reconciliation guard ──────────────────────────────────────────────────
    if len(bookings) != len(df_net):
        report.errors.append(
            f"Merge row count drifted from source: the Net file parsed to "
            f"{len(df_net)} bookings but the merged dataset has {len(bookings)}. "
            "The Net and Gross files could not be uniquely aligned — check for "
            "reordered or mismatched duplicate Res ID blocks between the two files."
        )

    pre_gross, post_gross = round(df_gross["Gross Sales"].sum(), 2), round(bookings["Gross Sales"].sum(), 2)
    pre_net, post_net = round(df_net["Net Sales"].sum(), 2), round(bookings["Net Sales"].sum(), 2)
    if abs(post_gross - pre_gross) > 0.01:
        report.errors.append(
            f"Gross Sales total changed during the Net/Gross merge (source file "
            f"total={pre_gross:.2f}, merged total={post_gross:.2f}). Refusing to "
            "trust these totals — investigate duplicate Res IDs before proceeding."
        )
    if abs(post_net - pre_net) > 0.01:
        report.errors.append(
            f"Net Sales total changed during the Net/Gross merge (source file "
            f"total={pre_net:.2f}, merged total={post_net:.2f}). Refusing to "
            "trust these totals — investigate duplicate Res IDs before proceeding."
        )

    return bookings


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_ems_files(
    net_path: Union[str, Path],
    gross_path: Union[str, Path],
    host_path: Union[str, Path],
    schema_map: Optional[dict[str, SchemaOverride]] = None,
) -> ParsedDataset:
    """
    Parse all three EMS export files and return a unified ParsedDataset.

    Parameters
    ----------
    net_path   : path to the "Net Sales by Booking" export
    gross_path : path to the "Gross Sales by Booking" export
    host_path  : path to the "Gross Sales by Host" export
    schema_map : optional {"net": {...}, "gross": {...}, "host": {...}}
                 overrides for files whose column headers don't match the
                 built-in aliases — each value maps canonical_field ->
                 column index (int) or header text to search for (str).

    Returns
    -------
    ParsedDataset with .bookings (merged), .host_summary, .validation
    """
    report = ValidationReport()
    schema_map = schema_map or {}

    # ── Read raw sheets ───────────────────────────────────────────────────────
    try:
        raw_net   = pd.read_excel(net_path,   sheet_name=0, header=None)
        raw_gross = pd.read_excel(gross_path, sheet_name=0, header=None)
        raw_host  = pd.read_excel(host_path,  sheet_name=0, header=None)
    except Exception as exc:
        report.errors.append(f"File read error: {exc}")
        return ParsedDataset(validation=report)

    period = _extract_reporting_period(raw_net)

    # ── Parse each file (schema resolved dynamically from headers) ───────────
    df_net,   net_col   = _parse_booking_sheet(raw_net,   report, schema_map.get("net"))
    df_gross, gross_col = _parse_booking_sheet(raw_gross, report, schema_map.get("gross"))
    df_host = _parse_host_sheet(raw_host, schema_map.get("host"))

    # The caller's net_path/gross_path labels are just a hint — the file's
    # actual header is what determines its role. If the two are swapped
    # (net_path points at the file containing "Gross Sales" and vice versa),
    # self-correct from the detected columns rather than trusting the label.
    if net_col == "Gross Sales" and gross_col == "Net Sales":
        report.warnings.append(
            "The file passed as the Net Sales booking file actually contains a "
            "Gross Sales column (and vice versa) — the two files appear to be "
            "swapped. Auto-corrected based on detected header content."
        )
        df_net, df_gross = df_gross, df_net
        net_col, gross_col = gross_col, net_col
    elif not df_net.empty and not df_gross.empty and net_col == gross_col:
        report.errors.append(
            f"Both booking files resolved to the same sales column ('{net_col}') "
            "— the Net and Gross files may be swapped, or one is missing its "
            "expected header. Check the uploaded files."
        )

    if not df_net.empty and "Net Sales" not in df_net.columns:
        report.errors.append(
            "The file provided as the Net Sales booking file does not contain a "
            "Net Sales column after parsing — aborting rather than compute on the "
            "wrong data."
        )
        return ParsedDataset(validation=report, reporting_period=period)
    if not df_gross.empty and "Gross Sales" not in df_gross.columns:
        report.errors.append(
            "The file provided as the Gross Sales booking file does not contain a "
            "Gross Sales column after parsing — aborting rather than compute on the "
            "wrong data."
        )
        return ParsedDataset(validation=report, reporting_period=period)

    # ── Cross-validate ────────────────────────────────────────────────────────
    _cross_validate(df_net, df_gross, report)

    # ── Merge net + gross (see _merge_net_and_gross for the reconciliation
    #    guard that catches duplicate-key fan-out before it reaches the UI) ──
    bookings = _merge_net_and_gross(df_net, df_gross, report)

    # ── Derived columns ───────────────────────────────────────────────────────
    if not bookings.empty:
        bookings["discount"] = (
            bookings["Gross Sales"] - bookings["Net Sales"]
        ).clip(lower=0)
        bookings["discount_pct"] = np.where(
            bookings["Gross Sales"] > 0,
            bookings["discount"] / bookings["Gross Sales"] * 100,
            0.0,
        )

        # Client segmentation: Host file's host_type is ground truth (used
        # ONLY as a name -> type lookup, never for dollar figures); keyword
        # heuristic is a fallback for any host not found in the Host file.
        bookings = _apply_host_type(bookings, df_host, report)

        # Fiscal year label (ASU FY: Jul–Jun)
        bookings["fiscal_year"] = bookings["start"].apply(
            lambda d: f"FY{(d.year + 1) % 100:02d}" if d.month >= 7 else f"FY{d.year % 100:02d}"
        )

    return ParsedDataset(
        bookings=bookings,
        host_summary=df_host,
        validation=report,
        reporting_period=period,
    )


def parse_single_file(
    filepath: Union[str, Path],
    file_role: str,
    schema_map: Optional[SchemaOverride] = None,
) -> dict:
    """
    Parse a single uploaded file. file_role ∈ {'net', 'gross', 'host'}.
    Returns a dict with 'data' (list of dicts) and 'validation'.
    """
    raw = pd.read_excel(filepath, sheet_name=0, header=None)
    report = ValidationReport()

    if file_role == "host":
        df = _parse_host_sheet(raw, schema_map)
    else:
        df, detected_col = _parse_booking_sheet(raw, report, schema_map)
        expected_col = "Net Sales" if file_role == "net" else "Gross Sales"
        if not df.empty and detected_col != expected_col:
            report.warnings.append(
                f"File was labeled '{file_role}' but its header indicates "
                f"'{detected_col}', not '{expected_col}'. Using the detected column."
            )
        if not df.empty:
            df["segment"] = df["host"].apply(_classify_host_segment)

    return {
        "data": df.to_dict(orient="records") if not df.empty else [],
        "rows": len(df),
        "validation": report.to_dict(),
    }
