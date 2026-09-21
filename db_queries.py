"""
Database query functions for all EHCD modules.
Projects, SG Offices, Task Management, Resolution Management.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg2.extras import RealDictCursor

from rbac import (
    is_superadmin,
    db_has_feature,
    FeatureID,
    accessible_sg_office_ids,
    accessible_task_ids,
    accessible_resolution_ids,
    user_can_access_sg_office,
    user_can_access_task,
    user_can_access_resolution,
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

def list_projects(conn, user_id: int, filters: dict = None) -> List[Dict[str, Any]]:
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
        # Manager: only projects they manage
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM project_management_project WHERE project_manager_id = %s ORDER BY id",
                (user_id,),
            )
            managed_ids = [r[0] for r in cur.fetchall()]
        if not managed_ids:
            return []
        conditions.append("p.id = ANY(%s)")
        params.append(managed_ids)

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

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY p.id"

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
    return result


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

        # Access check for non-admin
        if not all_projects:
            if project.get("project_manager_id") != user_id:
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

def list_sg_offices(conn, user_id: int, filters: dict = None) -> List[Dict[str, Any]]:
    """List SG offices the user can access."""
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
            return {"message": "You do not have access to any SG offices. Please contact your administrator to get access.", "data": []}
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
        return {"message": "No SG offices found matching your criteria.", "data": []}

    result = []
    for r in rows:
        item = dict(r)
        item["status_label"] = _status_label(item.get("status_en"))
        item["start_date"] = str(item["start_date"]) if item.get("start_date") else None
        item["end_date"] = str(item["end_date"]) if item.get("end_date") else None
        result.append(item)
    return result


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
# TASK MANAGEMENT
# ---------------------------------------------------------------------------

def list_tasks(conn, user_id: int, filters: dict = None) -> List[Dict[str, Any]]:
    """List tasks the user can access."""
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
            return {"message": "You do not have access to any tasks. Please contact your administrator to get access.", "data": []}
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
    return result


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
            SELECT id, task_id, task_name, task_name_ar, status_en, status_ar,
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

def list_resolutions(conn, user_id: int, filters: dict = None) -> List[Dict[str, Any]]:
    """List resolutions the user can access."""
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
            return {"message": "You do not have access to any resolutions. Please contact your administrator to get access.", "data": []}
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
    return result


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
                WHERE r.id = %s
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
