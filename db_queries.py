"""
Database query functions for all EHCD modules.
Projects, SG Offices, Task Management, Resolution Management.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg2.extras import RealDictCursor

from rbac import (
    is_superadmin,
    db_has_feature,
    FeatureID,
    accessible_project_ids,
    accessible_sg_office_ids,
    accessible_task_ids,
    accessible_resolution_ids,
    user_can_access_project,
    user_can_access_sg_office,
    user_can_access_task,
    user_can_access_resolution,
    has_sg_office_internal_access,
    has_sg_office_external_access,
)

# ---------------------------------------------------------------------------
# Status helpers (shared across modules)
# ---------------------------------------------------------------------------

STATUS_MAP_EN = {
    1: "In progress",
    2: "Completed",
    3: "Delayed",
    4: "On hold",
}
STATUS_MAP_AR = {
    1: "قيد التنفيذ",
    2: "مكتمل",
    3: "متأخر",
    4: "معلّق",
}

# SG Office external meetings — these integer codes aren't backed by a
# lookup table in the DB, so each mapping below was cross-verified against
# real UI screenshots joined back to their DB rows by requester/org name
# (not guessed). Where a code never appeared in an available screenshot,
# it's left out deliberately — _meeting_status_label() etc. fall back to
# "Status N" for anything not in these maps rather than invent a label.
MEETING_STATUS_MAP = {
    1: "New", 2: "Under Review", 3: "Confirmed",
    4: "Completed", 5: "Rescheduled", 6: "Cancelled",
}
# "Upcoming" (shown as its own summary count in the UI) = Confirmed + Rescheduled.
MEETING_UPCOMING_STATUSES = [3, 5]
MEETING_PRIORITY_MAP = {1: "High", 2: "Medium", 3: "Low"}
MEETING_REQUEST_TYPE_MAP = {1: "Meeting", 2: "Official Visit", 3: "Delegation Visit", 4: "Facility Visit"}
VISITOR_EMAIL_STATUS_MAP = {1: "Not Sent", 2: "Sent"}
READINESS_STATUS_MAP = {1: "Not Confirmed", 3: "Confirmed"}  # 2 still unconfirmed
FACILITY_STATUS_MAP = {1: "Not Confirmed", 3: "Confirmed"}  # 2 unconfirmed
FACILITY_TYPE_MAP = {
    5: "Room", 2: "Parking", 1: "Security Access",
    4: "Access Pass", 3: "Hospitality",
}
VENUE_MAP = {
    1: "Leadership Office - Meeting Room 1",
    2: "Leadership Office - Meeting Room 2",
    3: "Main Auditorium",
    4: "Reception Hall",
}

# DirectionItem enums — from the real source (common/enums.py), not
# cross-referenced guesses like the meeting ones above.
DIRECTION_ITEM_TYPE_MAP = {1: "Email Correspondence", 2: "Memo", 3: "Weekly Action"}
DIRECTION_CODE_PREFIX = {1: "SG", 2: "MEM", 3: "WA"}
DIRECTION_STATUS_MAP = {
    1: "Draft", 2: "New", 3: "Under Review", 4: "Awaiting H.E. Direction",
    5: "In Progress", 6: "Response Sent", 7: "Closed", 8: "Completed", 9: "Cancelled",
}
DIRECTION_FINISHED_STATUSES = (7, 8, 9)
DIRECTION_PRE_ASSIGNMENT_STATUSES = (2, 3, 4)
REQUIRED_DECISION_MAP = {
    1: "Approval", 2: "Signature", 3: "Nomination", 4: "Review and Endorsement",
    5: "Confirm Action Owners", 6: "Direction", 7: "For Information",
}
DIRECTION_OUTCOME_MAP = {
    1: "Approved", 2: "Approved with Amendments", 3: "Rejected",
    4: "Noted", 5: "More Information Requested",
}
CORRESPONDENCE_DIRECTION_MAP = {1: "Incoming", 2: "Outgoing"}
REMINDER_UNIT_MAP = {1: "Hours", 2: "Days"}
DIRECTION_EVENT_TYPE_MAP = {
    1: "Received", 2: "Summarized", 3: "AI Review Confirmed", 4: "Submitted for H.E. Direction",
    5: "Direction Recorded", 6: "Assigned", 7: "Reminder Sent", 8: "Escalated", 9: "Updated",
    10: "Note Added", 11: "Response Sent", 12: "Closed", 13: "Completed", 14: "Cancelled",
    15: "Reopened", 16: "Registered", 17: "Attachment Added", 18: "Attachment Removed",
    19: "Record Linked", 20: "Record Unlinked",
}


# This system is UAE-based (EHCD, Abu Dhabi); Postgres's own built-in
# today/now functions evaluate in the session's timezone, which is UTC
# here — the bare SQL keyword lags real UAE local time by a full calendar
# day during the ~4-hour window each night (UAE 00:00-03:59 = UTC 20:00-
# 23:59 the previous day), so "today"/"this week" filters could silently
# miss a meeting created "for today" in UAE time. Every date-boundary
# filter below uses this Dubai-local expression instead of that bare SQL
# keyword.
_TODAY_DUBAI = "(NOW() AT TIME ZONE 'Asia/Dubai')::date"


def _dubai_date(column: str) -> str:
    """SQL fragment: cast a timestamptz column to its Dubai-local calendar
    date, instead of `column::date` which casts using the session's (UTC)
    timezone and can land on the wrong day near the UAE day boundary."""
    return f"({column} AT TIME ZONE 'Asia/Dubai')::date"


def _label(val, mapping: dict, prefix: str) -> str:
    if val is None:
        return None
    return mapping.get(val, f"{prefix} {val}")


def _apply_fields(item: dict, fields, always_keep=()) -> dict:
    """Trim a result dict to just the requested fields — plus each one's
    resolved _label counterpart and a few always-kept identity fields — when
    the caller only asked about specific attributes. Returns the item
    unchanged when fields is empty, so full-detail callers are unaffected."""
    if not fields:
        return item
    keep = set(always_keep) | set(fields) | {f"{f}_label" for f in fields}
    return {k: v for k, v in item.items() if k in keep}


def _reorder_item(item: dict, priority_keys: tuple) -> dict:
    """Return item with priority_keys placed first, in the given order
    (skipping any not present), followed by all remaining keys in their
    original order — controls the column order a table renders in."""
    ordered = {k: item[k] for k in priority_keys if k in item}
    ordered.update((k, v) for k, v in item.items() if k not in ordered)
    return ordered


# Curated default view for meeting tables — mirrors the real frontend's
# "Meeting Requests" list page (Meeting/Requester/When & Where/Coordinator/
# Status) rather than dumping every column. Base names so _apply_fields
# auto-includes each one's resolved _label; still fully overridable via an
# explicit `fields` request for anything not in this default set.
MEETING_DEFAULT_FIELDS = (
    "meeting", "requester", "scheduled_date", "scheduled_time",
    "venue", "coordinator_name", "status", "facilities_summary",
    "readiness_summary",
)
MEETING_COLUMN_ORDER = (
    "meeting", "requester", "status_label", "scheduled_date",
    "scheduled_time", "venue_label", "priority_label", "coordinator_name",
    "facilities_summary", "readiness_summary", "visitor_arrival_summary",
)

# Preset for "meetings with incomplete visitor readiness" (visitors_not_ready
# filter) — one row per meeting; the Visitor column lists ONLY the visitors
# that aren't confirmed yet, not every visitor, and Status is a constant
# label since every returned row is incomplete by definition of the filter.
MEETING_INCOMPLETE_VISITORS_FIELDS = (
    "meeting", "incomplete_visitors_summary", "visitor_readiness_status",
)
# Preset for "meetings with pending facility requests" (not_ready filter) —
# same idea, Facility Request column lists only the not-yet-confirmed ones.
MEETING_PENDING_FACILITIES_FIELDS = (
    "meeting", "pending_facilities_summary", "facility_request_status",
)

# Curated default views for direction items — mirror the real frontend's
# Memo Register / Weekly Meeting Actions list pages. Memo and Weekly Action
# show different columns there, so the default is picked per-item from its
# own item_type; email_correspondence (and any unrecognized type) falls
# back to a generic set. Base names so _apply_fields auto-includes each
# one's resolved _label.
DIRECTION_ITEM_DEFAULT_FIELDS = {
    2: ("code", "date_received", "sender_name", "subject", "required_decision",
        "he_direction", "owner_name", "deadline", "status", "closed_at"),  # memo
    3: ("code", "subject", "source_meeting", "meeting_date", "owner_name",
        "deadline", "priority", "status", "reminder_label", "updated_at"),  # weekly_action
}
DIRECTION_ITEM_DEFAULT_FIELDS_GENERIC = (
    "code", "item_type_label", "subject", "sender_name", "date_received",
    "owner_name", "deadline", "status",
)
DIRECTION_ITEM_COLUMN_ORDER = (
    "code", "subject", "date_received", "sender_name", "source_meeting",
    "meeting_date", "required_decision_label", "he_direction", "owner_name",
    "deadline", "priority_label", "status_label", "reminder_label",
    "closed_at", "updated_at",
)

# Curated default view for the raw inbox — mirrors the frontend's email
# list (From/Subject/To-CC/Thread/Workflow/Summary/Received). "Thread" here
# is just thread_id (no per-thread message/unread aggregate exists yet).
EMAIL_DEFAULT_FIELDS = (
    "subject", "sender_name", "to_recipients_summary", "cc_recipients_summary",
    "received_datetime",
)
EMAIL_COLUMN_ORDER = (
    "subject", "sender_name", "to_recipients_summary", "cc_recipients_summary",
    "received_datetime",
)


def _format_recipients(recipients) -> str:
    """Flatten the raw Graph-API-style recipient JSON
    ([{"emailAddress": {"name": ..., "address": ...}}, ...]) into a short
    readable "Name <email>; Name2 <email2>" string — rendering the raw
    structure directly exploded into a huge nested sub-table per email,
    which is what made the emails table unreadably large."""
    if not recipients:
        return None
    parts = []
    for r in recipients:
        if not isinstance(r, dict):
            continue
        addr = r.get("emailAddress", r)
        name = addr.get("name") or ""
        email = addr.get("address") or ""
        if name and email and name != email:
            parts.append(f"{name} <{email}>")
        else:
            parts.append(name or email)
    return "; ".join(p for p in parts if p) or None


def _multi_word_ilike(column: str, text: str):
    """Build an (SQL fragment, params) pair that matches a column against
    EVERY word in `text`, in any order — e.g. searching "Emirates Foundation
    Youth Volunteering" still matches "Emirates Foundation — Youth
    Volunteering Programme" even though the em dash breaks a plain
    single-substring ILIKE. Used for free-text name/org/subject filters
    where a user's natural-language phrasing won't exactly match the DB's
    punctuation/spacing."""
    # Strip leading/trailing punctuation from each word (quotes, brackets,
    # trailing periods/colons, a standalone "-"/"–"/"—") before matching —
    # it adds no matching signal and is exactly how this breaks: a token
    # like "[FW:" (user wrapped a subject in brackets/quotes) or a plain
    # "-" where the DB has an en dash "–" never appears as a literal
    # substring in the real text, silently zeroing out an otherwise-correct
    # match on every other word once ANDed together.
    words = []
    for w in text.split():
        stripped = re.sub(r"^\W+|\W+$", "", w, flags=re.UNICODE)
        if stripped:
            words.append(stripped)
    if not words:
        return "TRUE", []
    clauses = [f"{column} ILIKE %s" for _ in words]
    params = [f"%{w}%" for w in words]
    return "(" + " AND ".join(clauses) + ")", params


def _status_label(val) -> str:
    try:
        return STATUS_MAP_EN.get(int(val), str(val))
    except (TypeError, ValueError):
        return str(val) if val else ""


def _fmt_jsonb(j: Any) -> str:
    if j is None:
        return ""
    if isinstance(j, (dict, list)):
        return json.dumps(j, ensure_ascii=False, indent=2)
    return str(j)


# ---------------------------------------------------------------------------
# PROJECTS
# ---------------------------------------------------------------------------

def list_projects(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List projects the user can access, with optional filters."""
    filters = filters or {}
    superadmin = is_superadmin(conn, user_id)
    all_projects = superadmin or db_has_feature(conn, user_id, FeatureID.ALL_PROJECTS)
    budget_ok = superadmin or db_has_feature(conn, user_id, FeatureID.BUDGET_INFO)

    query = """
        SELECT p.id, p.project_name_en, p.project_name_ar, p.status_en AS status, p.status_ar,
               p.start_date, p.end_date, p.project_manager_id,
               COALESCE(c.category_name_en, c.category_name_ar) AS category_name,
               u.full_name_en AS manager_name
        FROM project_management_project p
        LEFT JOIN project_management_projectcategory c ON c.id = p.project_category_id
        LEFT JOIN user_management_user u ON u.id = p.project_manager_id
    """
    conditions, params = [], []

    if not all_projects:
        # Manager or team member: only projects they manage or are listed
        # on the team for (see rbac.accessible_project_ids)
        accessible_ids = accessible_project_ids(conn, user_id)
        if not accessible_ids:
            return {
                "total_count": 0,
                "projects": []
            }
        conditions.append("p.id = ANY(%s)")
        params.append(accessible_ids)

    if filters.get("status"):
        status_val = filters["status"]
        # Try to map text status back to int
        reverse_map = {v.lower(): k for k, v in STATUS_MAP_EN.items()}
        status_int = reverse_map.get(status_val.lower().replace("_", " "))
        if status_int:
            conditions.append("p.status_en = %s")
            params.append(status_int)

    if filters.get("category"):
        conditions.append("(c.category_name_en ILIKE %s OR c.category_name_ar ILIKE %s)")
        params.extend([f"%{filters['category']}%", f"%{filters['category']}%"])

    if filters.get("project_manager"):
        conditions.append("u.full_name_en ILIKE %s")
        params.append(f"%{filters['project_manager']}%")

    if filters.get("start_date"):
        conditions.append("p.start_date::date = %s")
        params.append(filters["start_date"])

    if filters.get("end_date"):
        conditions.append("p.end_date::date = %s")
        params.append(filters["end_date"])

    # Month/year-only asks ("projects starting in September") have no exact
    # day to match — forcing them through start_date's exact-equality check
    # means the model has to guess a specific day, which almost never lands
    # on a real row (e.g. real September starts here are the 11th/13th/25th/
    # 30th, not the 1st) and silently returns zero results. These compare
    # only the parts actually being asked about.
    if filters.get("start_month"):
        conditions.append("EXTRACT(MONTH FROM p.start_date) = %s")
        params.append(filters["start_month"])

    if filters.get("start_year"):
        conditions.append("EXTRACT(YEAR FROM p.start_date) = %s")
        params.append(filters["start_year"])

    if filters.get("end_month"):
        conditions.append("EXTRACT(MONTH FROM p.end_date) = %s")
        params.append(filters["end_month"])

    if filters.get("end_year"):
        conditions.append("EXTRACT(YEAR FROM p.end_date) = %s")
        params.append(filters["end_year"])

    if filters.get("overdue"):
        # Deterministic end_date-vs-today comparison in SQL — without this,
        # the model was left to eyeball "has this project's due date
        # passed?" from the raw project list itself, and got it wrong (it
        # flagged a project ending 2027-08-31 as overdue on 2026-10-01).
        conditions.append("p.end_date::date < (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # "latest" means most recently started — p.id DESC (the default)
    # reflects insertion order, which doesn't match what a user means by
    # "the latest project" (e.g. a project entered into the system today
    # with a start date next month isn't "the latest").
    if filters.get("sort_by") == "latest":
        query += " ORDER BY p.start_date DESC NULLS LAST"
    else:
        query += " ORDER BY p.id DESC"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    # Batch-fetch budgets in one query instead of N+1
    budget_map = {}
    if budget_ok and rows:
        project_ids = [r["id"] for r in rows]
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT project_id, allocated_budget, spent_budget, budget_left
                FROM project_management_projectbudget
                WHERE project_id = ANY(%s)
            """, (project_ids,))
            for b in cur.fetchall():
                budget_map[b["project_id"]] = b

    result = []
    for r in rows:
        item = dict(r)
        item["status_en"] = _status_label(item.get("status"))
        item["start_date"] = str(item["start_date"]) if item.get("start_date") else None
        item["end_date"] = str(item["end_date"]) if item.get("end_date") else None
        if budget_ok:
            budget = budget_map.get(item["id"])
            if budget:
                item["allocated_budget"] = budget.get("allocated_budget")
                item["spent_budget"] = budget.get("spent_budget")
                item["budget_left"] = budget.get("budget_left")
        result.append(item)
    return {
    "total_count": len(result),
    "projects": result
    }


def _get_project_budget(conn, project_id: int) -> Optional[Dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT allocated_budget, spent_budget, budget_left
            FROM project_management_projectbudget
            WHERE project_id = %s LIMIT 1
        """, (project_id,))
        return cur.fetchone()


def get_project_details(conn, user_id: int, project_id: int = None,
                        project_name: str = None) -> Dict[str, Any]:
    """Get full project details by ID or name search."""
    superadmin = is_superadmin(conn, user_id)
    all_projects = superadmin or db_has_feature(conn, user_id, FeatureID.ALL_PROJECTS)
    budget_ok = superadmin or db_has_feature(conn, user_id, FeatureID.BUDGET_INFO)
    notes_ok = superadmin or db_has_feature(conn, user_id, FeatureID.NOTES)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if project_id:
            cur.execute("""
                SELECT p.*, COALESCE(c.category_name_en, c.category_name_ar) AS category_name,
                       u.full_name_en AS manager_name
                FROM project_management_project p
                LEFT JOIN project_management_projectcategory c ON c.id = p.project_category_id
                LEFT JOIN user_management_user u ON u.id = p.project_manager_id
                WHERE p.id = %s
            """, (project_id,))
        elif project_name:
            cur.execute("""
                SELECT p.*, COALESCE(c.category_name_en, c.category_name_ar) AS category_name,
                       u.full_name_en AS manager_name
                FROM project_management_project p
                LEFT JOIN project_management_projectcategory c ON c.id = p.project_category_id
                LEFT JOIN user_management_user u ON u.id = p.project_manager_id
                WHERE p.project_name_en ILIKE %s OR p.project_name_ar ILIKE %s
                LIMIT 1
            """, (f"%{project_name}%", f"%{project_name}%"))
        else:
            return {"error": "project_id or project_name is required"}

        project = cur.fetchone()
        if not project:
            return {"error": "Project not found"}

        pid = project["id"]

        # Access check for non-admin — manager or team member
        if not all_projects:
            if not user_can_access_project(conn, user_id, pid):
                return {"error": "Access denied to this project"}

        result = {
            "project": dict(project),
            "status_en": _status_label(project.get("status_en")),
        }

        # Budget
        if budget_ok:
            cur.execute("""
                SELECT allocated_budget, spent_budget, budget_left, created_at,
                       updaed_at AS updated_at
                FROM project_management_projectbudget WHERE project_id = %s LIMIT 1
            """, (pid,))
            result["budget"] = dict(cur.fetchone()) if cur.rowcount else None

        # Team
        cur.execute("""
            SELECT id, name_en, name_ar, designation_en, designation_ar, created_at, updated_at
            FROM project_management_teammember WHERE project_id = %s ORDER BY id
        """, (pid,))
        result["team"] = [dict(r) for r in cur.fetchall()]

        # Notes (only user's own notes)
        if notes_ok:
            cur.execute("""
                SELECT id, title, note, user_id, date, created_at, updated_at
                FROM project_management_projectnotes
                WHERE project_id = %s AND user_id = %s
                ORDER BY created_at DESC NULLS LAST, id DESC
            """, (pid, user_id))
            result["notes"] = [dict(r) for r in cur.fetchall()]

        # Format JSONB fields for readability
        for key in ["summary_heading_en", "summary_description_en", "progress_to_date_en",
                     "next_step_en", "next_step_due_date"]:
            val = project.get(key)
            if val:
                result["project"][key] = _fmt_jsonb(val)

    return result


# ---------------------------------------------------------------------------
# SG OFFICE
# ---------------------------------------------------------------------------

def list_sg_offices(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List SG offices the user can access. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}
    allowed_ids = accessible_sg_office_ids(conn, user_id)

    query = """
        SELECT s.id, s.sg_office_id, s.sg_office_name_en, s.sg_office_name_ar,
               s.start_date, s.end_date, s.status_en, s.status_ar,
               s.sg_office_description_en,
               c.category_name_en AS category_name,
               u.full_name_en AS manager_name
        FROM sg_office_sgoffice s
        LEFT JOIN sg_office_sgofficecategory c ON c.id = s.sg_office_category_id
        LEFT JOIN user_management_user u ON u.id = s.sg_office_manager_id
    """
    conditions, params = [], []

    if allowed_ids is not None:
        if not allowed_ids:
            return {"message": "You do not have access to any SG offices. Please contact your administrator to get access.", "total_count": 0, "data": []}
        conditions.append("s.id = ANY(%s)")
        params.append(allowed_ids)

    if filters.get("status"):
        status_val = filters["status"]
        reverse_map = {v.lower(): k for k, v in STATUS_MAP_EN.items()}
        status_int = reverse_map.get(status_val.lower().replace("_", " "))
        if status_int:
            conditions.append("s.status_en = %s")
            params.append(status_int)

    if filters.get("category_id"):
        conditions.append("s.sg_office_category_id = %s")
        params.append(filters["category_id"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.id"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        return {"message": "No SG offices found matching your criteria.", "total_count": 0, "data": []}

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _status_label(item.get("status_en"))
        item["start_date"] = str(item["start_date"]) if item.get("start_date") else None
        item["end_date"] = str(item["end_date"]) if item.get("end_date") else None
        result.append(item)
    return {"total_count": len(result), "data": result}


def get_sg_office_details(conn, user_id: int, sg_office_id: int = None,
                          sg_office_name: str = None) -> Dict[str, Any]:
    """Get full SG office details including budget, team, entities."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if sg_office_id:
            cur.execute("""
                SELECT s.*, c.category_name_en, c.category_name_ar,
                       u.full_name_en AS manager_name
                FROM sg_office_sgoffice s
                LEFT JOIN sg_office_sgofficecategory c ON c.id = s.sg_office_category_id
                LEFT JOIN user_management_user u ON u.id = s.sg_office_manager_id
                WHERE s.id = %s
            """, (sg_office_id,))
        elif sg_office_name:
            cur.execute("""
                SELECT s.*, c.category_name_en, c.category_name_ar,
                       u.full_name_en AS manager_name
                FROM sg_office_sgoffice s
                LEFT JOIN sg_office_sgofficecategory c ON c.id = s.sg_office_category_id
                LEFT JOIN user_management_user u ON u.id = s.sg_office_manager_id
                WHERE s.sg_office_name_en ILIKE %s OR s.sg_office_name_ar ILIKE %s
                LIMIT 1
            """, (f"%{sg_office_name}%", f"%{sg_office_name}%"))
        else:
            return {"error": "sg_office_id or sg_office_name is required"}

        office = cur.fetchone()
        if not office:
            return {"error": "SG Office not found"}

        oid = office["id"]

        if not user_can_access_sg_office(conn, user_id, oid):
            return {"error": "Access denied to this SG Office"}

        result = {"office": dict(office), "status_label": _status_label(office.get("status_en"))}

        # Budget
        cur.execute("""
            SELECT allocated_budget, spent_budget, budget_left, created_at,
                   updaed_at AS updated_at
            FROM sg_office_sgofficebudget WHERE sg_office_id = %s LIMIT 1
        """, (oid,))
        row = cur.fetchone()
        result["budget"] = dict(row) if row else None

        # Team
        cur.execute("""
            SELECT id, name_en, name_ar, designation_en, designation_ar, created_at, updated_at
            FROM sg_office_sgofficeteammember WHERE sg_office_id = %s ORDER BY id
        """, (oid,))
        result["team"] = [dict(r) for r in cur.fetchall()]

        # Entities
        cur.execute("""
            SELECT e.id, e.entity_name_en, e.entity_name_ar,
                   e.entity_acronym_en, e.entity_acronym_ar
            FROM sg_office_sgofficeentity e
            JOIN sg_office_sgoffice_sg_office_entities m ON m.sgofficeentity_id = e.id
            WHERE m.sgoffice_id = %s
        """, (oid,))
        result["entities"] = [dict(r) for r in cur.fetchall()]

        # Notes (user's own)
        cur.execute("""
            SELECT id, title, note, date, created_at, updated_at
            FROM sg_office_sgofficenotes
            WHERE sg_office_id = %s AND user_id = %s
            ORDER BY created_at DESC NULLS LAST, id DESC
        """, (oid, user_id))
        result["notes"] = [dict(r) for r in cur.fetchall()]

        # Format JSONB fields
        for key in ["summary_heading_en", "summary_description_en", "progress_to_date_en",
                     "next_step_en", "next_step_due_date"]:
            val = office.get(key)
            if val:
                result["office"][key] = _fmt_jsonb(val)

    return result


# ---------------------------------------------------------------------------
# SG OFFICE — INTERNAL DIRECTIONS (email correspondence)
#
# Read-only for now: the DB only has the raw synced-mailbox tables
# (sg_office_email / sg_office_emailthread / sg_office_emailattachment) —
# there's no table/column yet for the workflow layer the internal-tab spec
# describes (H.E. direction, assigned owner, deadline, status). Questions
# like "awaiting H.E. direction", "overdue directions", "waiting for a
# response", "due this week", "assigned to X" all describe that not-yet-
# built workflow layer, not the raw email. What IS real and queryable: the
# AI-generated `summary` column on both sg_office_email and
# sg_office_emailthread (the module's existing AI summary feature) — ask
# the model to summarize/read it directly off a fetched email or thread, no
# separate tool needed.
#
# RBAC: gated via rbac.has_sg_office_internal_access() / the
# "sg_office_internal" flag from get_user_access_flags(), applied at the
# tool-availability level in build_available_tools() (tools.py) — a user
# without role_and_access_feature id 11 ("Internal Meetings", or 13 "H.E.
# Briefings") never sees these tools offered at all, so no explicit check
# is needed inside the query functions themselves.
# ---------------------------------------------------------------------------

def list_sg_office_emails(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List SG Office internal email correspondence. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    # Excludes body_content deliberately — it's raw (often HTML) email body
    # and can be large; 29 matched rows with it included produced a ~1.6M
    # character tool result (~400K tokens), blowing past the model's
    # context window outright. summary/body_preview (both short, bounded
    # fields) are kept for scanning a list; get_sg_office_email_details
    # returns full body_content for one specific email/thread once the
    # list has narrowed down which one.
    query = """
        SELECT e.id, e.message_id, e.internet_message_id, e.subject, e.body_preview,
               e.summary, e.sender_name, e.sender_email, e.to_recipients, e.cc_recipients,
               e.bcc_recipients, e.reply_to, e.importance, e.has_attachments, e.is_read,
               e.is_draft, e.categories, e.flag, e.web_link, e.received_datetime,
               e.sent_datetime, e.thread_id, e.created_at, e.updated_at,
               u.full_name_en AS mailbox_owner_name,
               CASE WHEN EXISTS(
                   SELECT 1 FROM sg_office_directionitem d WHERE d.thread_id = e.thread_id
               ) THEN 'In Workflow' ELSE 'Not Started' END AS workflow_label
        FROM sg_office_email e
        LEFT JOIN user_management_user u ON u.id = e.mailbox_owner_id
    """
    conditions, params = [], []

    if filters.get("is_read") is not None:
        conditions.append("e.is_read = %s")
        params.append(filters["is_read"])

    if filters.get("is_draft") is not None:
        conditions.append("e.is_draft = %s")
        params.append(filters["is_draft"])

    if filters.get("flagged") is not None:
        if filters["flagged"]:
            conditions.append("e.flag->>'flagStatus' = 'flagged'")
        else:
            conditions.append("(e.flag->>'flagStatus' IS DISTINCT FROM 'flagged')")

    if filters.get("sender"):
        # Search sender_name and sender_email as ONE combined target, not
        # two separately-ANDed columns — a "Name <email>" style search (a
        # very natural way to specify a sender) has its name-words satisfied
        # by sender_name and its email satisfied by sender_email, but NEVER
        # both by either column alone, so the old OR-of-two-ANDs could never
        # match a combined name+email search even when it was exactly right.
        clause, p = _multi_word_ilike(
            "(e.sender_name || ' ' || COALESCE(e.sender_email, ''))", filters["sender"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("recipient"):
        c1, p1 = _multi_word_ilike("e.to_recipients::text", filters["recipient"])
        c2, p2 = _multi_word_ilike("e.cc_recipients::text", filters["recipient"])
        conditions.append(f"({c1} OR {c2})")
        params.extend(p1 + p2)

    if filters.get("subject"):
        clause, p = _multi_word_ilike("e.subject", filters["subject"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("duplicate_subject"):
        conditions.append("""
            e.subject IN (
                SELECT subject FROM sg_office_email
                WHERE subject IS NOT NULL
                GROUP BY subject HAVING COUNT(*) > 1
            )
        """)

    if filters.get("keyword"):
        # Matches subject OR body_preview — distinct from subject (subject
        # only) and body_contains (preview/content/summary, no subject).
        c1, p1 = _multi_word_ilike("e.subject", filters["keyword"])
        c2, p2 = _multi_word_ilike("e.body_preview", filters["keyword"])
        conditions.append(f"({c1} OR {c2})")
        params.extend(p1 + p2)

    if filters.get("body_contains"):
        c1, p1 = _multi_word_ilike("e.body_preview", filters["body_contains"])
        c2, p2 = _multi_word_ilike("e.body_content", filters["body_contains"])
        c3, p3 = _multi_word_ilike("e.summary", filters["body_contains"])
        conditions.append(f"({c1} OR {c2} OR {c3})")
        params.extend(p1 + p2 + p3)

    if filters.get("category"):
        conditions.append("e.categories::text ILIKE %s")
        params.append(f"%{filters['category']}%")

    if filters.get("importance"):
        conditions.append("e.importance = %s")
        params.append(filters["importance"])

    if filters.get("has_attachments") is not None:
        conditions.append("e.has_attachments = %s")
        params.append(filters["has_attachments"])

    if filters.get("thread_id"):
        conditions.append("e.thread_id = %s")
        params.append(filters["thread_id"])

    if filters.get("mailbox_owner"):
        clause, p = _multi_word_ilike("u.full_name_en", filters["mailbox_owner"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("received_after"):
        conditions.append("(e.received_datetime AT TIME ZONE 'Asia/Dubai')::date >= %s")
        params.append(filters["received_after"])

    if filters.get("received_before"):
        conditions.append("(e.received_datetime AT TIME ZONE 'Asia/Dubai')::date <= %s")
        params.append(filters["received_before"])

    if filters.get("older_than_days"):
        # Deterministic "open for more than N days" / "older than a week"
        # comparison in SQL — same reasoning as the projects `overdue`
        # filter: don't make the model do its own date math against
        # today's date, it gets it wrong.
        conditions.append("(e.received_datetime AT TIME ZONE 'Asia/Dubai')::date < ((NOW() AT TIME ZONE 'Asia/Dubai')::date - (%s || ' days')::interval)")
        params.append(filters["older_than_days"])

    if filters.get("received_today"):
        conditions.append("(e.received_datetime AT TIME ZONE 'Asia/Dubai')::date = (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if filters.get("received_yesterday"):
        conditions.append(
            "(e.received_datetime AT TIME ZONE 'Asia/Dubai')::date = "
            "(NOW() AT TIME ZONE 'Asia/Dubai')::date - INTERVAL '1 day'"
        )

    if filters.get("workflow_started") is not None:
        exists_clause = (
            "EXISTS(SELECT 1 FROM sg_office_directionitem d WHERE d.thread_id = e.thread_id)"
        )
        conditions.append(exists_clause if filters["workflow_started"] else f"NOT {exists_clause}")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if filters.get("sort_by") == "oldest":
        query += " ORDER BY e.received_datetime ASC"
    else:
        query += " ORDER BY e.received_datetime DESC"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["to_recipients_summary"] = _format_recipients(item.get("to_recipients"))
        item["cc_recipients_summary"] = _format_recipients(item.get("cc_recipients"))
        chosen_fields = filters.get("fields") or list(EMAIL_DEFAULT_FIELDS)
        item = _apply_fields(item, chosen_fields, always_keep=("sender_name", "subject"))
        item = _reorder_item(item, EMAIL_COLUMN_ORDER)
        result.append(item)

    return {"total_count": len(result), "data": result}


def get_sg_office_email_details(conn, user_id: int, email_id: int = None,
                                thread_id: int = None, latest_only: bool = False) -> Dict[str, Any]:
    """Get a single email (with its attachments), or a full thread — every
    message in it with its own attachments, or just the latest message in
    the thread if latest_only is set (answers "what's the latest response
    on this?" without the model having to scan the whole thread itself)."""
    if not email_id and not thread_id:
        return {"error": "email_id or thread_id is required"}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if email_id:
            cur.execute("""
                SELECT e.*, u.full_name_en AS mailbox_owner_name
                FROM sg_office_email e
                LEFT JOIN user_management_user u ON u.id = e.mailbox_owner_id
                WHERE e.id = %s
            """, (email_id,))
            email = cur.fetchone()
            if not email:
                return {"error": "Email not found"}
            email = dict(email)
            email["to_recipients"] = _format_recipients(email.get("to_recipients"))
            email["cc_recipients"] = _format_recipients(email.get("cc_recipients"))
            result = {"email": email}
            attachment_email_ids = [email["id"]]
        else:
            cur.execute("SELECT * FROM sg_office_emailthread WHERE id = %s", (thread_id,))
            thread = cur.fetchone()
            if not thread:
                return {"error": "Thread not found"}
            order = "DESC" if latest_only else "ASC"
            limit_clause = "LIMIT 1" if latest_only else ""
            cur.execute(f"""
                SELECT e.*, u.full_name_en AS mailbox_owner_name
                FROM sg_office_email e
                LEFT JOIN user_management_user u ON u.id = e.mailbox_owner_id
                WHERE e.thread_id = %s
                ORDER BY e.received_datetime {order}
                {limit_clause}
            """, (thread_id,))
            messages = [dict(r) for r in cur.fetchall()]
            for m in messages:
                m["to_recipients"] = _format_recipients(m.get("to_recipients"))
                m["cc_recipients"] = _format_recipients(m.get("cc_recipients"))
            if latest_only:
                result = {"thread": dict(thread), "latest_message": messages[0] if messages else None}
                attachment_email_ids = [messages[0]["id"]] if messages else []
            else:
                result = {"thread": dict(thread), "messages": messages}
                attachment_email_ids = [m["id"] for m in messages]

        if attachment_email_ids:
            cur.execute("""
                SELECT id, email_id, attachment_id, file_name, content_type, size, is_inline, blob_url
                FROM sg_office_emailattachment WHERE email_id = ANY(%s)
            """, (attachment_email_ids,))
            attachments = [dict(r) for r in cur.fetchall()]
            if email_id:
                result["attachments"] = attachments
            elif latest_only:
                result["latest_message"]["attachments"] = attachments
            else:
                by_email: Dict[int, list] = {}
                for a in attachments:
                    by_email.setdefault(a["email_id"], []).append(a)
                for m in result["messages"]:
                    m["attachments"] = by_email.get(m["id"], [])

    return result


# SG OFFICE — DIRECTION ITEMS (Email/Memo/Weekly Action unified workflow)
# RBAC: same "Internal Meetings" gate as the raw email tools above, applied
# at the tool-availability level — see the comment above list_sg_office_emails.

def list_sg_office_direction_items(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List Internal Direction items (emails/memos/weekly actions).
    Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    query = """
        SELECT d.*,
               o.full_name_en AS owner_name,
               cb.full_name_en AS created_by_name,
               drb.full_name_en AS direction_recorded_by_name,
               clb.full_name_en AS closed_by_name
        FROM sg_office_directionitem d
        LEFT JOIN user_management_user o ON o.id = d.owner_id
        LEFT JOIN user_management_user cb ON cb.id = d.created_by_id
        LEFT JOIN user_management_user drb ON drb.id = d.direction_recorded_by_id
        LEFT JOIN user_management_user clb ON clb.id = d.closed_by_id
    """
    conditions, params = [], []

    if filters.get("item_type"):
        reverse_map = {v.lower().replace(" ", "_"): k for k, v in DIRECTION_ITEM_TYPE_MAP.items()}
        it = reverse_map.get(str(filters["item_type"]).lower().replace(" ", "_"))
        if it:
            conditions.append("d.item_type = %s")
            params.append(it)

    if filters.get("code"):
        conditions.append("d.code = %s")
        params.append(filters["code"])

    if filters.get("status"):
        reverse_map = {v.lower(): k for k, v in DIRECTION_STATUS_MAP.items()}
        s = reverse_map.get(str(filters["status"]).lower().replace("_", " "))
        if s:
            conditions.append("d.status = %s")
            params.append(s)

    if filters.get("priority"):
        reverse_map = {v.lower(): k for k, v in MEETING_PRIORITY_MAP.items()}
        p = reverse_map.get(str(filters["priority"]).lower())
        if p:
            conditions.append("d.priority = %s")
            params.append(p)

    if filters.get("required_decision"):
        reverse_map = {v.lower(): k for k, v in REQUIRED_DECISION_MAP.items()}
        rd = reverse_map.get(str(filters["required_decision"]).lower().replace("_", " "))
        if rd:
            conditions.append("d.required_decision = %s")
            params.append(rd)

    if filters.get("direction_outcome"):
        reverse_map = {v.lower(): k for k, v in DIRECTION_OUTCOME_MAP.items()}
        do = reverse_map.get(str(filters["direction_outcome"]).lower().replace("_", " "))
        if do:
            conditions.append("d.direction_outcome = %s")
            params.append(do)

    if filters.get("correspondence_direction"):
        reverse_map = {v.lower(): k for k, v in CORRESPONDENCE_DIRECTION_MAP.items()}
        cd = reverse_map.get(str(filters["correspondence_direction"]).lower())
        if cd:
            conditions.append("d.correspondence_direction = %s")
            params.append(cd)

    if filters.get("cc_council_affairs") is not None:
        conditions.append("d.cc_council_affairs = %s")
        params.append(filters["cc_council_affairs"])

    if filters.get("subject_contains"):
        c1, p1 = _multi_word_ilike("d.subject", filters["subject_contains"])
        c2, p2 = _multi_word_ilike("d.description", filters["subject_contains"])
        c3, p3 = _multi_word_ilike("d.notes", filters["subject_contains"])
        conditions.append(f"({c1} OR {c2} OR {c3})")
        params.extend(p1 + p2 + p3)

    if filters.get("owner"):
        clause, p = _multi_word_ilike("o.full_name_en", filters["owner"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("created_by"):
        clause, p = _multi_word_ilike("cb.full_name_en", filters["created_by"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("sender_name"):
        clause, p = _multi_word_ilike("d.sender_name", filters["sender_name"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("sender_unit"):
        clause, p = _multi_word_ilike("d.sender_unit", filters["sender_unit"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("recipient_name"):
        clause, p = _multi_word_ilike("d.recipient_name", filters["recipient_name"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("reference"):
        conditions.append("d.reference ILIKE %s")
        params.append(f"%{filters['reference']}%")

    if filters.get("source_meeting"):
        clause, p = _multi_word_ilike("d.source_meeting", filters["source_meeting"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("date_received_after"):
        conditions.append("(d.date_received AT TIME ZONE 'Asia/Dubai')::date >= %s")
        params.append(filters["date_received_after"])

    if filters.get("meeting_date_on"):
        conditions.append("d.meeting_date = %s")
        params.append(filters["meeting_date_on"])

    if filters.get("deadline_before"):
        conditions.append("d.deadline <= %s")
        params.append(filters["deadline_before"])

    if filters.get("deadline_after"):
        conditions.append("d.deadline >= %s")
        params.append(filters["deadline_after"])

    if filters.get("overdue"):
        # Deadline passed and the item isn't in a finished state yet.
        conditions.append(
            f"d.deadline < (NOW() AT TIME ZONE 'Asia/Dubai')::date AND d.status NOT IN {DIRECTION_FINISHED_STATUSES}"
        )

    if filters.get("stalled"):
        # Per the enum's own PRE_ASSIGNMENT_STATUSES grouping: not yet
        # assigned/directed, and sitting for a few days.
        conditions.append(
            f"d.status IN {DIRECTION_PRE_ASSIGNMENT_STATUSES} "
            "AND (d.created_at AT TIME ZONE 'Asia/Dubai')::date < ((NOW() AT TIME ZONE 'Asia/Dubai')::date - INTERVAL '3 days')"
        )

    if filters.get("created_after"):
        conditions.append("(d.created_at AT TIME ZONE 'Asia/Dubai')::date >= %s")
        params.append(filters["created_after"])

    if filters.get("closed_after"):
        conditions.append("(d.closed_at AT TIME ZONE 'Asia/Dubai')::date >= %s")
        params.append(filters["closed_after"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if filters.get("sort_by") == "oldest":
        query += " ORDER BY d.created_at ASC"
    else:
        query += " ORDER BY d.created_at DESC"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["item_type_label"] = _label(item.get("item_type"), DIRECTION_ITEM_TYPE_MAP, "Type")
        item["status_label"] = _label(item.get("status"), DIRECTION_STATUS_MAP, "Status")
        item["priority_label"] = _label(item.get("priority"), MEETING_PRIORITY_MAP, "Priority")
        item["required_decision_label"] = _label(
            item.get("required_decision"), REQUIRED_DECISION_MAP, "Decision")
        item["direction_outcome_label"] = _label(
            item.get("direction_outcome"), DIRECTION_OUTCOME_MAP, "Outcome")
        item["correspondence_direction_label"] = _label(
            item.get("correspondence_direction"), CORRESPONDENCE_DIRECTION_MAP, "Direction")
        if item.get("reminder_value") and item.get("reminder_unit"):
            unit = _label(item.get("reminder_unit"), REMINDER_UNIT_MAP, "")
            item["reminder_label"] = f"Every {item['reminder_value']} {unit}"
        chosen_fields = filters.get("fields") or list(
            DIRECTION_ITEM_DEFAULT_FIELDS.get(item.get("item_type"), DIRECTION_ITEM_DEFAULT_FIELDS_GENERIC))
        item = _apply_fields(item, chosen_fields, always_keep=("code", "subject", "item_type_label"))
        item = _reorder_item(item, DIRECTION_ITEM_COLUMN_ORDER)
        result.append(item)

    return {"total_count": len(result), "data": result}


def get_sg_office_direction_item_details(conn, user_id: int, item_id: int = None,
                                         code: str = None) -> Dict[str, Any]:
    """Full detail for one Internal Direction item: the record itself,
    notes, attachments, the audit trail (events), and related items."""
    if not item_id and not code:
        return {"error": "item_id or code is required"}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if item_id:
            cur.execute("""
                SELECT d.*, o.full_name_en AS owner_name, cb.full_name_en AS created_by_name,
                       drb.full_name_en AS direction_recorded_by_name, clb.full_name_en AS closed_by_name
                FROM sg_office_directionitem d
                LEFT JOIN user_management_user o ON o.id = d.owner_id
                LEFT JOIN user_management_user cb ON cb.id = d.created_by_id
                LEFT JOIN user_management_user drb ON drb.id = d.direction_recorded_by_id
                LEFT JOIN user_management_user clb ON clb.id = d.closed_by_id
                WHERE d.id = %s
            """, (item_id,))
        else:
            cur.execute("""
                SELECT d.*, o.full_name_en AS owner_name, cb.full_name_en AS created_by_name,
                       drb.full_name_en AS direction_recorded_by_name, clb.full_name_en AS closed_by_name
                FROM sg_office_directionitem d
                LEFT JOIN user_management_user o ON o.id = d.owner_id
                LEFT JOIN user_management_user cb ON cb.id = d.created_by_id
                LEFT JOIN user_management_user drb ON drb.id = d.direction_recorded_by_id
                LEFT JOIN user_management_user clb ON clb.id = d.closed_by_id
                WHERE d.code = %s
            """, (code,))
        item = cur.fetchone()
        if not item:
            return {"error": "Direction item not found"}

        item = dict(item)
        iid = item["id"]
        item["item_type_label"] = _label(item.get("item_type"), DIRECTION_ITEM_TYPE_MAP, "Type")
        item["status_label"] = _label(item.get("status"), DIRECTION_STATUS_MAP, "Status")
        item["priority_label"] = _label(item.get("priority"), MEETING_PRIORITY_MAP, "Priority")
        item["required_decision_label"] = _label(
            item.get("required_decision"), REQUIRED_DECISION_MAP, "Decision")
        item["direction_outcome_label"] = _label(
            item.get("direction_outcome"), DIRECTION_OUTCOME_MAP, "Outcome")
        item["correspondence_direction_label"] = _label(
            item.get("correspondence_direction"), CORRESPONDENCE_DIRECTION_MAP, "Direction")

        cur.execute("""
            SELECT n.*, a.full_name_en AS author_name
            FROM sg_office_directionitemnote n
            LEFT JOIN user_management_user a ON a.id = n.author_id
            WHERE n.item_id = %s ORDER BY n.created_at
        """, (iid,))
        notes = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT at.*, u.full_name_en AS uploaded_by_name
            FROM sg_office_directionitemattachment at
            LEFT JOIN user_management_user u ON u.id = at.uploaded_by_id
            WHERE at.item_id = %s ORDER BY at.created_at
        """, (iid,))
        attachments = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT ev.*, a.full_name_en AS actor_name
            FROM sg_office_directionitemevent ev
            LEFT JOIN user_management_user a ON a.id = ev.actor_id
            WHERE ev.item_id = %s ORDER BY ev.created_at
        """, (iid,))
        audit_trail = []
        for r in cur.fetchall():
            e = dict(r)
            e["event_type_label"] = _label(e.get("event_type"), DIRECTION_EVENT_TYPE_MAP, "Event")
            e["from_status_label"] = _label(e.get("from_status"), DIRECTION_STATUS_MAP, "Status")
            e["to_status_label"] = _label(e.get("to_status"), DIRECTION_STATUS_MAP, "Status")
            audit_trail.append(e)

        cur.execute("""
            SELECT d2.id, d2.code, d2.subject, d2.item_type
            FROM sg_office_directionitem_related_items r
            JOIN sg_office_directionitem d2 ON d2.id = r.to_directionitem_id
            WHERE r.from_directionitem_id = %s
            UNION
            SELECT d2.id, d2.code, d2.subject, d2.item_type
            FROM sg_office_directionitem_related_items r
            JOIN sg_office_directionitem d2 ON d2.id = r.from_directionitem_id
            WHERE r.to_directionitem_id = %s
        """, (iid, iid))
        related = []
        for r in cur.fetchall():
            rr = dict(r)
            rr["item_type_label"] = _label(rr.get("item_type"), DIRECTION_ITEM_TYPE_MAP, "Type")
            related.append(rr)

    return {
        "item": item,
        "notes": notes,
        "attachments": attachments,
        "audit_trail": audit_trail,
        "related_items": related,
    }


# ---------------------------------------------------------------------------
# SG OFFICE — EXTERNAL MEETINGS, VISITORS & FACILITIES (Theyab's workspace)
#
# Six tables: sg_office_meetingrequest (core), sg_office_meetingrequest_participants
# (M2M to users), sg_office_meetingrequestfacility (prep tasks),
# sg_office_meetingrequestvisitor (readiness), sg_office_meetingoutcome
# (post-meeting notes), sg_office_meetingrequeststatushistory (the Audit
# Trail shown in the UI). All six are real and populated — unlike Internal
# Directions, this module HAS its full workflow schema (status, priority,
# assignees, deadlines all exist as real columns), so filters here aren't
# limited by missing schema the way the email ones are.
#
# RBAC: gated via rbac.has_sg_office_external_access() / the
# "sg_office_external" flag, applied at the tool-availability level — a
# user without role_and_access_feature id 12 ("Meeting Requests", or 13
# "H.E. Briefings") never sees these tools offered at all.
# ---------------------------------------------------------------------------

def list_sg_office_meetings(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List external meeting/visit requests. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    query = """
        SELECT m.*,
               co.full_name_en AS coordinator_name,
               cb.full_name_en AS created_by_name,
               cf.full_name_en AS confirmed_by_name,
               (SELECT ARRAY_AGG(DISTINCT f.facility ORDER BY f.facility)
                FROM sg_office_meetingrequestfacility f
                WHERE f.meeting_request_id = m.id) AS facility_codes,
               (SELECT COUNT(*) FROM sg_office_meetingrequestfacility f
                WHERE f.meeting_request_id = m.id) AS facilities_total,
               (SELECT COUNT(*) FROM sg_office_meetingrequestfacility f
                WHERE f.meeting_request_id = m.id AND f.status = 3) AS facilities_ready,
               (SELECT COUNT(*) FROM sg_office_meetingrequestvisitor v
                WHERE v.meeting_request_id = m.id) AS visitors_total,
               (SELECT COUNT(*) FROM sg_office_meetingrequestvisitor v
                WHERE v.meeting_request_id = m.id AND v.readiness_status = 3) AS visitors_ready,
               (SELECT ARRAY_AGG(v.name) FROM sg_office_meetingrequestvisitor v
                WHERE v.meeting_request_id = m.id AND v.readiness_status != 3) AS incomplete_visitor_names,
               (SELECT ARRAY_AGG(DISTINCT f.facility ORDER BY f.facility)
                FROM sg_office_meetingrequestfacility f
                WHERE f.meeting_request_id = m.id AND f.status != 3) AS pending_facility_codes
        FROM sg_office_meetingrequest m
        LEFT JOIN user_management_user co ON co.id = m.coordinator_id
        LEFT JOIN user_management_user cb ON cb.id = m.created_by_id
        LEFT JOIN user_management_user cf ON cf.id = m.confirmed_by_id
    """
    conditions, params = [], []

    if filters.get("status"):
        status_val = str(filters["status"]).lower().replace("_", " ")
        reverse_map = {v.lower(): k for k, v in MEETING_STATUS_MAP.items()}
        if status_val == "upcoming":
            conditions.append("m.status = ANY(%s)")
            params.append(MEETING_UPCOMING_STATUSES)
        elif status_val in reverse_map:
            conditions.append("m.status = %s")
            params.append(reverse_map[status_val])

    if filters.get("priority"):
        reverse_map = {v.lower(): k for k, v in MEETING_PRIORITY_MAP.items()}
        p = reverse_map.get(str(filters["priority"]).lower())
        if p:
            conditions.append("m.priority = %s")
            params.append(p)

    if filters.get("request_type"):
        reverse_map = {v.lower(): k for k, v in MEETING_REQUEST_TYPE_MAP.items()}
        rt = reverse_map.get(str(filters["request_type"]).lower().replace("_", " "))
        if rt:
            conditions.append("m.request_type = %s")
            params.append(rt)

    if filters.get("requester"):
        clause, p = _multi_word_ilike("m.requester", filters["requester"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("organization"):
        clause, p = _multi_word_ilike("m.organization", filters["organization"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("email"):
        conditions.append("m.email ILIKE %s")
        params.append(f"%{filters['email']}%")

    if filters.get("purpose_contains"):
        clause1, p1 = _multi_word_ilike("m.purpose", filters["purpose_contains"])
        clause2, p2 = _multi_word_ilike("m.notes", filters["purpose_contains"])
        conditions.append(f"({clause1} OR {clause2})")
        params.extend(p1 + p2)

    if filters.get("coordinator"):
        clause, p = _multi_word_ilike("co.full_name_en", filters["coordinator"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("created_by"):
        clause, p = _multi_word_ilike("cb.full_name_en", filters["created_by"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("visitor_email_status"):
        reverse_map = {v.lower(): k for k, v in VISITOR_EMAIL_STATUS_MAP.items()}
        ves = reverse_map.get(str(filters["visitor_email_status"]).lower().replace("_", " "))
        if ves:
            conditions.append("m.visitor_email_status = %s")
            params.append(ves)

    if filters.get("is_confirmed") is not None:
        if filters["is_confirmed"]:
            conditions.append("m.confirmed_at IS NOT NULL")
        else:
            conditions.append("m.confirmed_at IS NULL")

    if filters.get("notify_stakeholders") is not None:
        conditions.append("m.notify_stakeholders = %s")
        params.append(filters["notify_stakeholders"])

    if filters.get("scheduled_after"):
        conditions.append("m.scheduled_date >= %s")
        params.append(filters["scheduled_after"])

    if filters.get("scheduled_before"):
        conditions.append("m.scheduled_date <= %s")
        params.append(filters["scheduled_before"])

    if filters.get("scheduled_on"):
        conditions.append("m.scheduled_date = %s")
        params.append(filters["scheduled_on"])

    if filters.get("scheduled_today"):
        # Deterministic — don't make the model compute "today" itself.
        conditions.append("m.scheduled_date = (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if filters.get("scheduled_this_week"):
        # ISO week (Monday-Sunday) containing today, computed in SQL so the
        # model never has to work out week boundaries itself.
        conditions.append(
            "m.scheduled_date BETWEEN date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date)::date "
            "AND (date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date) + INTERVAL '6 days')::date"
        )

    if filters.get("scheduled_tomorrow"):
        conditions.append("m.scheduled_date = (NOW() AT TIME ZONE 'Asia/Dubai')::date + INTERVAL '1 day'")

    if filters.get("scheduled_this_month"):
        conditions.append(
            "m.scheduled_date BETWEEN date_trunc('month', (NOW() AT TIME ZONE 'Asia/Dubai')::date)::date "
            "AND (date_trunc('month', (NOW() AT TIME ZONE 'Asia/Dubai')::date) + INTERVAL '1 month - 1 day')::date"
        )

    if filters.get("not_ready"):
        # At least one facility prep task isn't Confirmed yet (status != 3).
        conditions.append(
            "m.id IN (SELECT f.meeting_request_id FROM sg_office_meetingrequestfacility f WHERE f.status != 3)"
        )

    if filters.get("visitors_not_ready"):
        # At least one visitor's readiness isn't confirmed (status = 1 —
        # the only value this system confirms means "Not Confirmed"). This
        # is MEETING-level (one row per meeting), unlike
        # list_sg_office_meeting_visitors which is one row per visitor —
        # use this for "which MEETINGS have incomplete visitor readiness".
        conditions.append(
            "m.id IN (SELECT v.meeting_request_id FROM sg_office_meetingrequestvisitor v WHERE v.readiness_status = 1)"
        )

    if filters.get("duplicate_organization"):
        conditions.append("""
            m.organization IN (
                SELECT organization FROM sg_office_meetingrequest
                WHERE organization IS NOT NULL
                GROUP BY organization HAVING COUNT(*) > 1
            )
        """)

    if filters.get("scheduled_on_weekday"):
        # ISODOW computed in SQL — never let the model work out which
        # calendar date a weekday name falls on, it gets this wrong.
        weekday_map = {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
                       "friday": 5, "saturday": 6, "sunday": 7}
        dow = weekday_map.get(str(filters["scheduled_on_weekday"]).lower())
        if dow:
            conditions.append("EXTRACT(ISODOW FROM m.scheduled_date) = %s AND m.scheduled_date >= (NOW() AT TIME ZONE 'Asia/Dubai')::date")
            params.append(dow)

    if filters.get("upcoming_only"):
        # For "upcoming X" where X is a specific status (e.g. "upcoming
        # confirmed") — status=upcoming alone means Confirmed+Rescheduled,
        # not what's wanted here. This restricts to future-dated regardless
        # of status, combine with a specific status filter above.
        conditions.append("m.scheduled_date >= (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if filters.get("created_after"):
        conditions.append("(m.created_at AT TIME ZONE 'Asia/Dubai')::date >= %s")
        params.append(filters["created_after"])

    if filters.get("older_than_days"):
        conditions.append("(m.created_at AT TIME ZONE 'Asia/Dubai')::date < ((NOW() AT TIME ZONE 'Asia/Dubai')::date - (%s || ' days')::interval)")
        params.append(filters["older_than_days"])

    if filters.get("stalled"):
        # "Stalled" per the Theyab workspace spec = still New/Under Review
        # (nothing confirmed, no response) and sitting for a while — 3 days
        # is this function's own default for "a while"; pass older_than_days
        # instead for a custom threshold on an unconfirmed request.
        conditions.append(
            "m.status IN (1, 2) AND (m.created_at AT TIME ZONE 'Asia/Dubai')::date < ((NOW() AT TIME ZONE 'Asia/Dubai')::date - INTERVAL '3 days')"
        )

    if filters.get("participant"):
        clause, p = _multi_word_ilike("u.full_name_en", filters["participant"])
        conditions.append(f"""m.id IN (
            SELECT mp.meetingrequest_id FROM sg_office_meetingrequest_participants mp
            JOIN user_management_user u ON u.id = mp.user_id
            WHERE {clause}
        )""")
        params.extend(p)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if filters.get("sort_by") == "oldest":
        query += " ORDER BY m.created_at ASC"
    elif filters.get("sort_by") == "soonest":
        # Soonest upcoming scheduled_date first — use for "next meeting"
        # style questions. Combine with upcoming_only so past dates don't
        # sort to the front.
        query += " ORDER BY m.scheduled_date ASC NULLS LAST"
    else:
        query += " ORDER BY m.created_at DESC"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _label(item.get("status"), MEETING_STATUS_MAP, "Status")
        item["priority_label"] = _label(item.get("priority"), MEETING_PRIORITY_MAP, "Priority")
        item["request_type_label"] = _label(item.get("request_type"), MEETING_REQUEST_TYPE_MAP, "Type")
        item["venue_label"] = _label(item.get("venue"), VENUE_MAP, "Venue")
        item["visitor_email_status_label"] = _label(
            item.get("visitor_email_status"), VISITOR_EMAIL_STATUS_MAP, "Status")
        facility_codes = item.pop("facility_codes", None) or []
        item["facilities_summary"] = (
            ", ".join(_label(c, FACILITY_TYPE_MAP, "Facility") for c in facility_codes)
            if facility_codes else None
        )
        # Same "ready/total" formula as get_sg_office_meeting_details's
        # readiness_summary (Confirmed facilities + Confirmed visitors, out
        # of both totals) — exposed here too so a list of meetings (e.g.
        # "meetings with pending facility requests") can show each one's
        # readiness at a glance instead of needing a drill-down per meeting.
        f_total, f_ready = item.pop("facilities_total", 0) or 0, item.pop("facilities_ready", 0) or 0
        v_total, v_ready = item.pop("visitors_total", 0) or 0, item.pop("visitors_ready", 0) or 0
        total = f_total + v_total
        item["readiness_summary"] = f"{f_ready + v_ready}/{total}" if total else None

        incomplete_names = item.pop("incomplete_visitor_names", None) or []
        item["incomplete_visitors_summary"] = ", ".join(incomplete_names) if incomplete_names else None
        item["visitor_readiness_status"] = "Incomplete" if incomplete_names else "Complete"

        pending_codes = item.pop("pending_facility_codes", None) or []
        item["pending_facilities_summary"] = (
            ", ".join(_label(c, FACILITY_TYPE_MAP, "Facility") for c in pending_codes)
            if pending_codes else None
        )
        item["facility_request_status"] = "Pending" if pending_codes else "Confirmed"

        # "Meeting" reads better than "Organization" as a column header for
        # these tables — the underlying `organization` filter param is
        # unchanged, this only renames what's shown.
        item["meeting"] = item.pop("organization", None)

        # Query-specific presets: a meeting-level "incomplete readiness" or
        # "pending facilities" question gets ONE lean table (Meeting +
        # only-the-incomplete-items + a status word) instead of the general
        # default — no repeated meeting rows, no irrelevant columns.
        if filters.get("fields"):
            chosen_fields, keep = filters["fields"], ("requester", "meeting", "scheduled_date")
        elif filters.get("visitors_not_ready"):
            chosen_fields, keep = list(MEETING_INCOMPLETE_VISITORS_FIELDS), ("meeting",)
        elif filters.get("not_ready"):
            chosen_fields, keep = list(MEETING_PENDING_FACILITIES_FIELDS), ("meeting",)
        else:
            chosen_fields, keep = list(MEETING_DEFAULT_FIELDS), ("requester", "meeting", "scheduled_date")
        item = _apply_fields(item, chosen_fields, always_keep=keep)
        item = _reorder_item(item, MEETING_COLUMN_ORDER)
        result.append(item)

    return {"total_count": len(result), "data": result}


def get_sg_office_meeting_details(conn, user_id: int, meeting_request_id: int,
                                  view: str = None) -> Dict[str, Any]:
    """Full detail for one meeting/visit request: the request itself,
    participants, facility prep tasks, visitor readiness, the post-meeting
    outcome (if any), and the full status-change audit trail.

    `view` narrows the result to exactly what a specific question type
    needs, instead of always returning everything:
    - "readiness": one combined table, one row per visitor or facility
      prep task, under a shared Visitor/Readiness/Arrived/Facility Request/
      Status schema — no meeting name/org column (the answer text names
      the meeting), no repeated meeting rows.
    - "facility_status": one table, Facility Request/Status only.
    - "meeting_info": meeting name/date/time/duration/venue/coordinator +
      participants, no requester/status/facilities/visitors/outcome/audit.
    - None (default): the full record, unchanged.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT m.*,
                   co.full_name_en AS coordinator_name,
                   cb.full_name_en AS created_by_name,
                   cf.full_name_en AS confirmed_by_name
            FROM sg_office_meetingrequest m
            LEFT JOIN user_management_user co ON co.id = m.coordinator_id
            LEFT JOIN user_management_user cb ON cb.id = m.created_by_id
            LEFT JOIN user_management_user cf ON cf.id = m.confirmed_by_id
            WHERE m.id = %s
        """, (meeting_request_id,))
        meeting = cur.fetchone()
        if not meeting:
            return {"error": "Meeting request not found"}

        meeting = dict(meeting)
        meeting["status_label"] = _label(meeting.get("status"), MEETING_STATUS_MAP, "Status")
        meeting["priority_label"] = _label(meeting.get("priority"), MEETING_PRIORITY_MAP, "Priority")
        meeting["request_type_label"] = _label(meeting.get("request_type"), MEETING_REQUEST_TYPE_MAP, "Type")
        meeting["venue_label"] = _label(meeting.get("venue"), VENUE_MAP, "Venue")
        meeting["visitor_email_status_label"] = _label(
            meeting.get("visitor_email_status"), VISITOR_EMAIL_STATUS_MAP, "Status")
        meeting["meeting"] = meeting.pop("organization", None)
        # Full-detail view still uses the curated set as its base record —
        # the nested sections below (participants/facilities/visitors/
        # outcome/audit_trail) are where the deeper, specifically-asked-for
        # detail actually lives.
        meeting = _apply_fields(
            meeting, list(MEETING_DEFAULT_FIELDS) + ["purpose", "request_type", "duration_minutes"],
            always_keep=("requester", "meeting", "scheduled_date"))
        meeting = _reorder_item(meeting, MEETING_COLUMN_ORDER)

        cur.execute("""
            SELECT u.id, u.full_name_en, u.full_name_ar, u.email, u.designation
            FROM sg_office_meetingrequest_participants mp
            JOIN user_management_user u ON u.id = mp.user_id
            WHERE mp.meetingrequest_id = %s
        """, (meeting_request_id,))
        participants = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT f.*, a.full_name_en AS assigned_to_name, u.full_name_en AS updated_by_name
            FROM sg_office_meetingrequestfacility f
            LEFT JOIN user_management_user a ON a.id = f.assigned_to_id
            LEFT JOIN user_management_user u ON u.id = f.updated_by_id
            WHERE f.meeting_request_id = %s
            ORDER BY f.id
        """, (meeting_request_id,))
        facilities = []
        for r in cur.fetchall():
            f = dict(r)
            f["status_label"] = _label(f.get("status"), FACILITY_STATUS_MAP, "Status")
            f["priority_label"] = _label(f.get("priority"), MEETING_PRIORITY_MAP, "Priority")
            f["facility_label"] = _label(f.get("facility"), FACILITY_TYPE_MAP, "Facility")
            facilities.append(f)

        cur.execute("""
            SELECT v.*, ru.full_name_en AS readiness_updated_by_name,
                   am.full_name_en AS arrival_marked_by_name
            FROM sg_office_meetingrequestvisitor v
            LEFT JOIN user_management_user ru ON ru.id = v.readiness_updated_by_id
            LEFT JOIN user_management_user am ON am.id = v.arrival_marked_by_id
            WHERE v.meeting_request_id = %s
            ORDER BY v.id
        """, (meeting_request_id,))
        visitors = []
        for r in cur.fetchall():
            v = dict(r)
            v["readiness_status_label"] = _label(v.get("readiness_status"), READINESS_STATUS_MAP, "Status")
            v["arrived"] = v.get("arrived_at") is not None
            visitors.append(v)

        cur.execute("""
            SELECT o.*, cb.full_name_en AS completed_by_name
            FROM sg_office_meetingoutcome o
            LEFT JOIN user_management_user cb ON cb.id = o.completed_by_id
            WHERE o.meeting_request_id = %s
        """, (meeting_request_id,))
        outcome_row = cur.fetchone()
        outcome = dict(outcome_row) if outcome_row else None

        cur.execute("""
            SELECT h.*, u.full_name_en AS changed_by_name
            FROM sg_office_meetingrequeststatushistory h
            LEFT JOIN user_management_user u ON u.id = h.changed_by_id
            WHERE h.meeting_request_id = %s
            ORDER BY h.changed_at
        """, (meeting_request_id,))
        audit_trail = []
        for r in cur.fetchall():
            h = dict(r)
            h["from_status_label"] = _label(h.get("from_status"), MEETING_STATUS_MAP, "Status")
            h["to_status_label"] = _label(h.get("to_status"), MEETING_STATUS_MAP, "Status")
            audit_trail.append(h)

        # Readiness = Confirmed (status 3) facility-prep tasks + Confirmed
        # (readiness_status 3) visitors, out of the total of both — matches
        # the real frontend's "X/Y Ready" readiness bar exactly (confirmed
        # by cross-referencing a real meeting: 2 facilities both unconfirmed
        # + 7 visitors with 6 confirmed = 6/9, matching the UI's own count).
        ready_count = (
            sum(1 for f in facilities if f.get("status") == 3)
            + sum(1 for v in visitors if v.get("readiness_status") == 3)
        )
        total_count = len(facilities) + len(visitors)
        arrived_count = sum(1 for v in visitors if v.get("arrived"))
        # Kept on the "meeting" record (not as a new top-level key) so the
        # table renderer still treats this whole result as one primary
        # record + sections, instead of falling back to a less readable
        # generic layout.
        meeting["readiness_summary"] = f"{ready_count}/{total_count}" if total_count else None
        meeting["visitor_arrival_summary"] = f"{arrived_count}/{len(visitors)} arrived" if visitors else None

    if view == "readiness":
        # One shared column schema for every row regardless of origin, so
        # they render as ONE table (a mismatched schema per row would make
        # the renderer split them into separate tables).
        checklist = []
        for v in visitors:
            checklist.append({
                "visitor": v.get("name"),
                "readiness": v.get("readiness_status_label"),
                "arrived": "Arrived" if v.get("arrived") else "Not Arrived",
                "facility_request": None,
                "status": None,
            })
        for f in facilities:
            checklist.append({
                "visitor": None,
                "readiness": None,
                "arrived": None,
                "facility_request": f.get("facility_label"),
                "status": f.get("status_label"),
            })
        return {"total_count": len(checklist), "data": checklist}

    if view == "facility_status":
        rows = [{"facility_request": f.get("facility_label"), "status": f.get("status_label")} for f in facilities]
        return {"total_count": len(rows), "data": rows}

    if view == "meeting_info":
        info = _apply_fields(
            meeting,
            ["meeting", "scheduled_date", "scheduled_time", "duration_minutes", "venue", "coordinator_name"],
            always_keep=("meeting",))
        info = _reorder_item(info, MEETING_COLUMN_ORDER)
        return {"meeting": info, "participants": participants}

    return {
        "meeting": meeting,
        "participants": participants,
        "facilities": facilities,
        "visitors": visitors,
        "outcome": outcome,
        "audit_trail": audit_trail,
    }


def list_sg_office_meeting_facilities(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """Cross-meeting facility-prep tracker (Theyab's 'Preparation' board) —
    e.g. 'which facility requests are unassigned', 'what's overdue for prep'.
    Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    query = """
        SELECT f.*, m.requester, m.organization, m.scheduled_date, m.status AS meeting_status,
               a.full_name_en AS assigned_to_name, u.full_name_en AS updated_by_name
        FROM sg_office_meetingrequestfacility f
        JOIN sg_office_meetingrequest m ON m.id = f.meeting_request_id
        LEFT JOIN user_management_user a ON a.id = f.assigned_to_id
        LEFT JOIN user_management_user u ON u.id = f.updated_by_id
    """
    conditions, params = [], []

    if filters.get("organization"):
        clause, p = _multi_word_ilike("m.organization", filters["organization"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("requester"):
        clause, p = _multi_word_ilike("m.requester", filters["requester"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("status"):
        reverse_map = {v.lower(): k for k, v in FACILITY_STATUS_MAP.items()}
        s = reverse_map.get(str(filters["status"]).lower().replace("_", " "))
        if s:
            conditions.append("f.status = %s")
            params.append(s)

    if filters.get("priority"):
        reverse_map = {v.lower(): k for k, v in MEETING_PRIORITY_MAP.items()}
        p = reverse_map.get(str(filters["priority"]).lower())
        if p:
            conditions.append("f.priority = %s")
            params.append(p)

    if filters.get("facility"):
        reverse_map = {v.lower(): k for k, v in FACILITY_TYPE_MAP.items()}
        ft = reverse_map.get(str(filters["facility"]).lower().replace("_", " "))
        if ft:
            conditions.append("f.facility = %s")
            params.append(ft)

    if filters.get("unassigned") is not None:
        if filters["unassigned"]:
            conditions.append("f.assigned_to_id IS NULL")
        else:
            conditions.append("f.assigned_to_id IS NOT NULL")

    if filters.get("assigned_to"):
        clause, p = _multi_word_ilike("a.full_name_en", filters["assigned_to"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("overdue"):
        conditions.append("f.due_at IS NOT NULL AND f.due_at < NOW() AND f.status != 1")

    if filters.get("due_before"):
        conditions.append("(f.due_at AT TIME ZONE 'Asia/Dubai')::date <= %s")
        params.append(filters["due_before"])

    if filters.get("scheduled_today"):
        conditions.append("m.scheduled_date = (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if filters.get("scheduled_this_week"):
        conditions.append(
            "m.scheduled_date BETWEEN date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date)::date "
            "AND (date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date) + INTERVAL '6 days')::date"
        )

    if filters.get("meeting_request_id"):
        conditions.append("f.meeting_request_id = %s")
        params.append(filters["meeting_request_id"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY f.due_at NULLS LAST, f.id"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _label(item.get("status"), FACILITY_STATUS_MAP, "Status")
        item["priority_label"] = _label(item.get("priority"), MEETING_PRIORITY_MAP, "Priority")
        item["facility_label"] = _label(item.get("facility"), FACILITY_TYPE_MAP, "Facility")
        item["meeting_status_label"] = _label(item.get("meeting_status"), MEETING_STATUS_MAP, "Status")
        result.append(item)

    return {"total_count": len(result), "data": result}


def list_sg_office_meeting_visitors(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """Cross-meeting visitor readiness tracker (Theyab's readiness board) —
    e.g. 'which visitors haven't arrived', 'whose readiness isn't confirmed'.
    Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    query = """
        SELECT v.*, m.requester, m.organization, m.scheduled_date, m.status AS meeting_status
        FROM sg_office_meetingrequestvisitor v
        JOIN sg_office_meetingrequest m ON m.id = v.meeting_request_id
    """
    conditions, params = [], []

    if filters.get("organization"):
        clause, p = _multi_word_ilike("m.organization", filters["organization"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("requester"):
        clause, p = _multi_word_ilike("m.requester", filters["requester"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("readiness_status"):
        reverse_map = {v.lower(): k for k, v in READINESS_STATUS_MAP.items()}
        rs = reverse_map.get(str(filters["readiness_status"]).lower().replace("_", " "))
        if rs:
            conditions.append("v.readiness_status = %s")
            params.append(rs)

    if filters.get("arrived") is not None:
        if filters["arrived"]:
            conditions.append("v.arrived_at IS NOT NULL")
        else:
            conditions.append("v.arrived_at IS NULL")

    if filters.get("email_failed") is not None:
        conditions.append("v.email_failed = %s")
        params.append(filters["email_failed"])

    if filters.get("name"):
        clause, p = _multi_word_ilike("v.name", filters["name"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("scheduled_today"):
        conditions.append("m.scheduled_date = (NOW() AT TIME ZONE 'Asia/Dubai')::date")

    if filters.get("scheduled_this_week"):
        conditions.append(
            "m.scheduled_date BETWEEN date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date)::date "
            "AND (date_trunc('week', (NOW() AT TIME ZONE 'Asia/Dubai')::date) + INTERVAL '6 days')::date"
        )

    if filters.get("meeting_request_id"):
        conditions.append("v.meeting_request_id = %s")
        params.append(filters["meeting_request_id"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY m.scheduled_date NULLS LAST, v.id"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["readiness_status_label"] = _label(item.get("readiness_status"), READINESS_STATUS_MAP, "Status")
        item["meeting_status_label"] = _label(item.get("meeting_status"), MEETING_STATUS_MAP, "Status")
        item["arrived"] = item.get("arrived_at") is not None
        result.append(item)

    return {"total_count": len(result), "data": result}


def list_sg_office_meeting_outcomes(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """Cross-meeting outcomes tracker — e.g. 'which completed meetings need
    a follow-up'. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}

    query = """
        SELECT o.*, m.requester, m.organization, m.scheduled_date, m.status AS meeting_status,
               cb.full_name_en AS completed_by_name
        FROM sg_office_meetingoutcome o
        JOIN sg_office_meetingrequest m ON m.id = o.meeting_request_id
        LEFT JOIN user_management_user cb ON cb.id = o.completed_by_id
    """
    conditions, params = [], []

    if filters.get("organization"):
        clause, p = _multi_word_ilike("m.organization", filters["organization"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("requester"):
        clause, p = _multi_word_ilike("m.requester", filters["requester"])
        conditions.append(clause)
        params.extend(p)

    if filters.get("follow_up_required") is not None:
        conditions.append("o.follow_up_required = %s")
        params.append(filters["follow_up_required"])

    if filters.get("completed_after"):
        conditions.append("o.date_completed >= %s")
        params.append(filters["completed_after"])

    if filters.get("meeting_request_id"):
        conditions.append("o.meeting_request_id = %s")
        params.append(filters["meeting_request_id"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY o.date_completed DESC NULLS LAST, o.id DESC"

    if filters.get("limit"):
        query += " LIMIT %s"
        params.append(filters["limit"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["meeting_status_label"] = _label(item.get("meeting_status"), MEETING_STATUS_MAP, "Status")
        result.append(item)

    return {"total_count": len(result), "data": result}


# ---------------------------------------------------------------------------
# TASK MANAGEMENT
# ---------------------------------------------------------------------------

def list_tasks(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List tasks the user can access with optional filters. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}
    allowed_ids = accessible_task_ids(conn, user_id)

    query = """
        SELECT t.id, t.task_name, t.task_name_ar,
               t.status_en, t.status_ar, t.date_of_request,
               t.requires_presentation_to_main_council,
               t.presentation_readiness, t.got_presented,
               t.status_detail_en,
               u.full_name_en AS owner_name,
               c.category_name_en AS category_name,
               cm.committee_name_en AS committee_name
        FROM task_management_task t
        LEFT JOIN user_management_user u ON u.id = t.owner_id
        LEFT JOIN project_management_projectcategory c ON c.id = t.category_id
        LEFT JOIN task_management_committee cm ON cm.id = t.proposed_committee_id
    """
    conditions, params = [], []

    if allowed_ids is not None:
        if not allowed_ids:
            return {"message": "You do not have access to any tasks. Please contact your administrator to get access.", "total_count": 0, "data": []}
        conditions.append("t.id = ANY(%s)")
        params.append(allowed_ids)

    if filters.get("status"):
        status_val = filters["status"]
        reverse_map = {v.lower(): k for k, v in STATUS_MAP_EN.items()}
        status_int = reverse_map.get(status_val.lower().replace("_", " "))
        if status_int:
            conditions.append("t.status_en = %s")
            params.append(status_int)

    if filters.get("entity_name"):
        conditions.append("""t.entity_id IN (
            SELECT id FROM project_management_projectentity
            WHERE entity_name_en ILIKE %s OR entity_name_ar ILIKE %s
        )""")
        params.extend([f"%{filters['entity_name']}%", f"%{filters['entity_name']}%"])

    if filters.get("requires_presentation") is not None:
        conditions.append("t.requires_presentation_to_main_council = %s")
        params.append(filters["requires_presentation"])

    if filters.get("request_date"):
        conditions.append("(t.date_of_request AT TIME ZONE 'Asia/Dubai')::date = %s")
        params.append(filters["request_date"])

    if filters.get("advisor"):
        conditions.append("""t.id IN (
            SELECT task_id FROM task_management_taskadvisor
            WHERE name_en ILIKE %s OR name_ar ILIKE %s
        )""")
        params.extend([f"%{filters['advisor']}%", f"%{filters['advisor']}%"])

    # Filter by subtask (top-level tasks only if requested)
    if filters.get("top_level_only"):
        conditions.append("t.subtask_id IS NULL")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY t.id"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _status_label(item.get("status_en"))
        item["date_of_request"] = str(item["date_of_request"]) if item.get("date_of_request") else None
        result.append(item)
    return {"total_count": len(result), "data": result}


def get_task_details(conn, user_id: int, task_id: int = None,
                     task_name: str = None) -> Dict[str, Any]:
    """Get full task details including advisors, committee, subtasks."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if task_id:
            cur.execute("""
                SELECT t.*,
                       u.full_name_en AS owner_name,
                       c.category_name_en AS category_name,
                       cm.committee_name_en AS committee_name,
                       e.entity_name_en AS entity_name
                FROM task_management_task t
                LEFT JOIN user_management_user u ON u.id = t.owner_id
                LEFT JOIN project_management_projectcategory c ON c.id = t.category_id
                LEFT JOIN task_management_committee cm ON cm.id = t.proposed_committee_id
                LEFT JOIN project_management_projectentity e ON e.id = t.entity_id
                WHERE t.id = %s
            """, (task_id,))
        elif task_name:
            cur.execute("""
                SELECT t.*,
                       u.full_name_en AS owner_name,
                       c.category_name_en AS category_name,
                       cm.committee_name_en AS committee_name,
                       e.entity_name_en AS entity_name
                FROM task_management_task t
                LEFT JOIN user_management_user u ON u.id = t.owner_id
                LEFT JOIN project_management_projectcategory c ON c.id = t.category_id
                LEFT JOIN task_management_committee cm ON cm.id = t.proposed_committee_id
                LEFT JOIN project_management_projectentity e ON e.id = t.entity_id
                WHERE t.task_name ILIKE %s OR t.task_name_ar ILIKE %s
                LIMIT 1
            """, (f"%{task_name}%", f"%{task_name}%"))
        else:
            return {"error": "task_id or task_name is required"}

        task = cur.fetchone()
        if not task:
            return {"error": "Task not found"}

        tid = task["id"]

        if not user_can_access_task(conn, user_id, tid):
            return {"error": "Access denied to this task"}

        result = {"task": dict(task), "status_label": _status_label(task.get("status_en"))}

        # Advisors
        cur.execute("""
            SELECT id, name_en, name_ar, designation_en, designation_ar, created_at
            FROM task_management_taskadvisor WHERE task_id = %s ORDER BY id
        """, (tid,))
        result["advisors"] = [dict(r) for r in cur.fetchall()]

        # Committee members (if task has a proposed committee)
        if task.get("proposed_committee_id"):
            cur.execute("""
                SELECT cm.id, u.full_name_en AS member_name, u.email, u.designation
                FROM task_management_committeemember cm
                JOIN user_management_user u ON u.id = cm.user_id
                WHERE cm.committee_id = %s ORDER BY cm.id
            """, (task["proposed_committee_id"],))
            result["committee_members"] = [dict(r) for r in cur.fetchall()]

        # Subtasks
        cur.execute("""
            SELECT id, task_name, task_name_ar, status_en, status_ar,
                   date_of_request, status_detail_en
            FROM task_management_task WHERE subtask_id = %s ORDER BY id
        """, (tid,))
        subtasks = cur.fetchall()
        result["subtasks"] = []
        for st in subtasks:
            d = dict(st)
            d["status_label"] = _status_label(d.get("status_en"))
            result["subtasks"].append(d)

    return result


# ---------------------------------------------------------------------------
# RESOLUTION MANAGEMENT
# ---------------------------------------------------------------------------

def list_resolutions(conn, user_id: int, filters: dict = None) -> Dict[str, Any]:
    """List resolutions the user can access. Returns {"total_count": N, "data": [...]}."""
    filters = filters or {}
    allowed_ids = accessible_resolution_ids(conn, user_id)

    query = """
        SELECT r.id, r.resolution_id, r.resolution_topic_en, r.resolution_topic_ar,
               r.resolution_status, r.deadline_completion, r.meeting_date,
               r.year_of_resolution,
               u.full_name_en AS owner_name,
               e.entity_name_en AS entity_name
        FROM resolution_management_resolution r
        LEFT JOIN user_management_user u ON u.id = r.resolution_owner_id
        LEFT JOIN project_management_projectentity e ON e.id = r.responsible_entity_id
    """
    conditions, params = [], []

    if allowed_ids is not None:
        if not allowed_ids:
            return {"message": "You do not have access to any resolutions. Please contact your administrator to get access.", "total_count": 0, "data": []}
        conditions.append("r.id = ANY(%s)")
        params.append(allowed_ids)

    if filters.get("status"):
        status_val = filters["status"]
        reverse_map = {v.lower(): k for k, v in STATUS_MAP_EN.items()}
        status_int = reverse_map.get(status_val.lower().replace("_", " "))
        if status_int:
            conditions.append("r.resolution_status = %s")
            params.append(status_int)

    if filters.get("year"):
        conditions.append("r.year_of_resolution = %s")
        params.append(filters["year"])

    if filters.get("entity_name"):
        conditions.append("""r.responsible_entity_id IN (
            SELECT id FROM project_management_projectentity
            WHERE entity_name_en ILIKE %s OR entity_name_ar ILIKE %s
        )""")
        params.extend([f"%{filters['entity_name']}%", f"%{filters['entity_name']}%"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY r.id"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _status_label(item.get("resolution_status"))
        item["deadline_completion"] = str(item["deadline_completion"]) if item.get("deadline_completion") else None
        item["meeting_date"] = str(item["meeting_date"]) if item.get("meeting_date") else None
        result.append(item)
    return {"total_count": len(result), "data": result}


def get_resolution_details(conn, user_id: int, resolution_id: int = None,
                           resolution_topic: str = None) -> Dict[str, Any]:
    """Get full resolution details including details, committees, supporting team."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if resolution_id:
            cur.execute("""
                SELECT r.*,
                       u.full_name_en AS owner_name,
                       e.entity_name_en AS entity_name,
                       cb.full_name_en AS created_by_name
                FROM resolution_management_resolution r
                LEFT JOIN user_management_user u ON u.id = r.resolution_owner_id
                LEFT JOIN project_management_projectentity e ON e.id = r.responsible_entity_id
                LEFT JOIN user_management_user cb ON cb.id = r.created_by_id
                WHERE r.resolution_id = %s
            """, (resolution_id,))
        elif resolution_topic:
            cur.execute("""
                SELECT r.*,
                       u.full_name_en AS owner_name,
                       e.entity_name_en AS entity_name,
                       cb.full_name_en AS created_by_name
                FROM resolution_management_resolution r
                LEFT JOIN user_management_user u ON u.id = r.resolution_owner_id
                LEFT JOIN project_management_projectentity e ON e.id = r.responsible_entity_id
                LEFT JOIN user_management_user cb ON cb.id = r.created_by_id
                WHERE r.resolution_topic_en ILIKE %s OR r.resolution_topic_ar ILIKE %s
                LIMIT 1
            """, (f"%{resolution_topic}%", f"%{resolution_topic}%"))
        else:
            return {"error": "resolution_id or resolution_topic is required"}

        resolution = cur.fetchone()
        if not resolution:
            return {"error": "Resolution not found"}

        rid = resolution["id"]

        if not user_can_access_resolution(conn, user_id, rid):
            return {"error": "Access denied to this resolution"}

        result = {
            "resolution": dict(resolution),
            "status_label": _status_label(resolution.get("resolution_status")),
        }

        # Details
        cur.execute("""
            SELECT id, detail_en, detail_ar, created_at
            FROM resolution_management_resolutiondetail
            WHERE resolution_id = %s ORDER BY id
        """, (rid,))
        result["details"] = [dict(r) for r in cur.fetchall()]

        # Committees
        cur.execute("""
            SELECT rc.id, c.committee_name_en, c.committee_name_ar
            FROM resolution_management_resolutioncommittee rc
            JOIN task_management_committee c ON c.id = rc.committee_id
            WHERE rc.resolution_id = %s ORDER BY rc.id
        """, (rid,))
        result["committees"] = [dict(r) for r in cur.fetchall()]

        # Supporting team
        cur.execute("""
            SELECT st.id, u.full_name_en AS member_name, u.email,
                   u.designation, u.department
            FROM resolution_management_resolutionsupportingteam st
            JOIN user_management_user u ON u.id = st.user_id
            WHERE st.resolution_id = %s ORDER BY st.id
        """, (rid,))
        result["supporting_team"] = [dict(r) for r in cur.fetchall()]

    return result
