"""
RBAC (Role-Based Access Control) module for EHCD Chatbot.
Handles user roles, features, and ownership-based access for all modules.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from psycopg2 import sql
from psycopg2.extras import RealDictCursor

USER_ROLES_TABLE = os.getenv("RBAC_USER_ROLES_TABLE", "user_management_user_roles")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")


class FeatureID:
    USER_MANAGEMENT    = 3
    BUDGET_INFO        = 4
    EDUCATION_DASH     = 5
    ALL_PROJECTS       = 6
    NOTES              = 7
    PROJECT_DOCS       = 8
    AI_CHATBOT         = 9
    SG_OFFICE_INTERNAL = 10  # Internal Directions tab (email/memos/weekly
                             # actions) — viewable by H.E., Shamma, and
                             # Theyab per the SG Office internal workflow
                             # spec.
    SG_OFFICE_EXTERNAL = 11  # External Meetings/Visitors/Facilities tab.
                             # Neither of these two has a row in
                             # role_and_access_feature / role_and_access_role_features
                             # in prod yet (confirmed 2026-10: catalog only
                             # has ids 1-9) — grant via that table like any
                             # other feature once these roles are defined;
                             # until then db_has_feature() always returns
                             # False for them, which is exactly why neither
                             # gate is actually called yet (see db_queries.py).


def is_superadmin(conn, user_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT is_superuser FROM user_management_user WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return bool(row and row[0])


def db_has_feature(conn, user_id: int, feature_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM user_management_user_roles ur
            JOIN role_and_access_role_features rf ON rf.role_id = ur.role_id
            WHERE ur.user_id = %s AND rf.feature_id = %s
            LIMIT 1
        """, (user_id, feature_id))
        return cur.fetchone() is not None


def fetch_user_roles_features(conn, user_id: int) -> Tuple[List[str], Set[str]]:
    roles, feats = [], set()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql.SQL("""
            SELECT ur.role_id
            FROM {} ur
            WHERE ur.user_id = %s
        """).format(sql.Identifier(DB_SCHEMA, USER_ROLES_TABLE)), (user_id,))
        roles = [str(row["role_id"]) for row in cur.fetchall()]

        cur.execute(sql.SQL("""
            SELECT f.id as feature_id, f.feature_name
            FROM {} ur
            JOIN role_and_access_role_features rf ON rf.role_id = ur.role_id
            JOIN role_and_access_feature f ON f.id = rf.feature_id
            WHERE ur.user_id = %s
        """).format(sql.Identifier(DB_SCHEMA, USER_ROLES_TABLE)), (user_id,))
        feats = {row["feature_name"].lower() for row in cur.fetchall()}
    return roles, feats


def fetch_user_profile(conn, user_id: int) -> Dict[str, Any]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql.SQL("""
            SELECT u.full_name_en, u.full_name_ar, u.email, u.department,
                   u.designation, u.contact_no
            FROM {}.user_management_user u
            WHERE u.id = %s
        """).format(sql.Identifier(DB_SCHEMA)), (user_id,))
        return cur.fetchone() or {}


def fetch_all_users(conn) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, full_name_en, full_name_ar, email, department, designation,
                   contact_no, is_active, is_staff, is_superuser, created_at, updated_at
            FROM user_management_user
            ORDER BY id
        """)
        return cur.fetchall() or []


def has_all_projects(conn, user_id: int) -> bool:
    return db_has_feature(conn, user_id, FeatureID.ALL_PROJECTS)


def has_budget(conn, user_id: int) -> bool:
    return db_has_feature(conn, user_id, FeatureID.BUDGET_INFO)


def has_education_access(conn, user_id: int) -> bool:
    return db_has_feature(conn, user_id, FeatureID.EDUCATION_DASH)


def has_notes(conn, user_id: int) -> bool:
    return db_has_feature(conn, user_id, FeatureID.NOTES)


def has_user_management(conn, user_id: int) -> bool:
    return db_has_feature(conn, user_id, FeatureID.USER_MANAGEMENT)


def has_sg_office_internal_access(conn, user_id: int) -> bool:
    """H.E., Shamma, and Theyab all see the Internal Directions tab (email
    correspondence, memos, weekly actions) — a flat view/no-view gate, not
    ownership-based like projects/tasks/resolutions, since all three named
    roles see the same shared data."""
    return is_superadmin(conn, user_id) or db_has_feature(conn, user_id, FeatureID.SG_OFFICE_INTERNAL)


def has_sg_office_external_access(conn, user_id: int) -> bool:
    """H.E., Shamma, and Theyab all see the External Meetings/Visitors/
    Facilities tab — same flat view/no-view gate as the internal one."""
    return is_superadmin(conn, user_id) or db_has_feature(conn, user_id, FeatureID.SG_OFFICE_EXTERNAL)


# ---------------------------------------------------------------------------
# Ownership-based access helpers for new modules
# ---------------------------------------------------------------------------

def _is_admin_or_super(conn, user_id: int) -> bool:
    return is_superadmin(conn, user_id) or db_has_feature(conn, user_id, FeatureID.ALL_PROJECTS)


_column_check_cache: Dict[str, bool] = {}

def _table_has_column(conn, table_name: str, column_name: str) -> bool:
    cache_key = f"{table_name}.{column_name}"
    if cache_key in _column_check_cache:
        return _column_check_cache[cache_key]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
            """,
            (DB_SCHEMA, table_name, column_name),
        )
        result = cur.fetchone() is not None
    _column_check_cache[cache_key] = result
    return result


def accessible_project_ids(conn, user_id: int) -> Optional[List[int]]:
    """Return list of project IDs the user can see. None means all.
    Mirrors accessible_sg_office_ids: project_management_teammember has no
    user_id column today, so team-member access falls back to matching the
    member's stored name against the user's own full_name_en — fragile
    (typos, duplicate names) but it's what the equivalent SG-office code
    already does in production, and upgrades automatically if a user_id
    column is ever added to this table."""
    if _is_admin_or_super(conn, user_id):
        return None
    with conn.cursor() as cur:
        if _table_has_column(conn, "project_management_teammember", "user_id"):
            cur.execute("""
                SELECT id FROM project_management_project WHERE project_manager_id = %s
                UNION
                SELECT project_id
                FROM project_management_teammember
                WHERE user_id = %s
            """, (user_id, user_id))
        else:
            cur.execute("""
                SELECT id FROM project_management_project WHERE project_manager_id = %s
                UNION
                SELECT project_id
                FROM project_management_teammember
                WHERE name_en IN (
                    SELECT full_name_en FROM user_management_user WHERE id = %s
                )
            """, (user_id, user_id))
        return [r[0] for r in cur.fetchall()]


def user_can_access_project(conn, user_id: int, project_id: int) -> bool:
    if _is_admin_or_super(conn, user_id):
        return True
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM project_management_project
            WHERE id = %s AND project_manager_id = %s
            LIMIT 1
        """, (project_id, user_id))
        if cur.fetchone():
            return True
        if _table_has_column(conn, "project_management_teammember", "user_id"):
            cur.execute("""
                SELECT 1 FROM project_management_teammember
                WHERE project_id = %s AND user_id = %s
                LIMIT 1
            """, (project_id, user_id))
        else:
            cur.execute("""
                SELECT 1 FROM project_management_teammember
                WHERE project_id = %s
                  AND name_en IN (
                      SELECT full_name_en FROM user_management_user WHERE id = %s
                  )
                LIMIT 1
            """, (project_id, user_id))
        if cur.fetchone():
            return True
    return False


def accessible_sg_office_ids(conn, user_id: int) -> Optional[List[int]]:
    """Return list of sg_office IDs the user can see. None means all."""
    if _is_admin_or_super(conn, user_id):
        return None
    with conn.cursor() as cur:
        if _table_has_column(conn, "sg_office_sgofficeteammember", "user_id"):
            cur.execute("""
                SELECT id FROM sg_office_sgoffice WHERE sg_office_manager_id = %s
                UNION
                SELECT sg_office_id
                FROM sg_office_sgofficeteammember
                WHERE user_id = %s
            """, (user_id, user_id))
        else:
            cur.execute("""
                SELECT id FROM sg_office_sgoffice WHERE sg_office_manager_id = %s
                UNION
                SELECT sg_office_id
                FROM sg_office_sgofficeteammember
                WHERE name_en IN (
                    SELECT full_name_en FROM user_management_user WHERE id = %s
                )
            """, (user_id, user_id))
        return [r[0] for r in cur.fetchall()]


def user_can_access_sg_office(conn, user_id: int, sg_office_id: int) -> bool:
    if _is_admin_or_super(conn, user_id):
        return True
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM sg_office_sgoffice
            WHERE id = %s AND sg_office_manager_id = %s
            LIMIT 1
        """, (sg_office_id, user_id))
        if cur.fetchone():
            return True
        if _table_has_column(conn, "sg_office_sgofficeteammember", "user_id"):
            cur.execute("""
                SELECT 1 FROM sg_office_sgofficeteammember
                WHERE sg_office_id = %s AND user_id = %s
                LIMIT 1
            """, (sg_office_id, user_id))
        else:
            cur.execute("""
                SELECT 1 FROM sg_office_sgofficeteammember
                WHERE sg_office_id = %s
                  AND name_en IN (
                      SELECT full_name_en FROM user_management_user WHERE id = %s
                  )
                LIMIT 1
            """, (sg_office_id, user_id))
        if cur.fetchone():
            return True
    return False


def accessible_task_ids(conn, user_id: int) -> Optional[List[int]]:
    """Return list of task IDs the user can see. None means all."""
    if _is_admin_or_super(conn, user_id):
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM task_management_task WHERE owner_id = %s
            UNION
            SELECT id FROM task_management_task WHERE created_by_id = %s
            UNION
            SELECT t.id FROM task_management_task t
            JOIN task_management_committeemember cm ON cm.committee_id = t.proposed_committee_id
            WHERE cm.user_id = %s
        """, (user_id, user_id, user_id))
        return [r[0] for r in cur.fetchall()]


def user_can_access_task(conn, user_id: int, task_id: int) -> bool:
    if _is_admin_or_super(conn, user_id):
        return True
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM task_management_task
            WHERE id = %s AND (owner_id = %s OR created_by_id = %s)
            LIMIT 1
        """, (task_id, user_id, user_id))
        if cur.fetchone():
            return True
        cur.execute("""
            SELECT 1 FROM task_management_committeemember cm
            JOIN task_management_task t ON t.proposed_committee_id = cm.committee_id
            WHERE t.id = %s AND cm.user_id = %s
            LIMIT 1
        """, (task_id, user_id))
        return cur.fetchone() is not None


def accessible_resolution_ids(conn, user_id: int) -> Optional[List[int]]:
    """Return list of resolution IDs the user can see. None means all."""
    if _is_admin_or_super(conn, user_id):
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM resolution_management_resolution
            WHERE resolution_owner_id = %s OR created_by_id = %s
            UNION
            SELECT resolution_id FROM resolution_management_resolutionsupportingteam
            WHERE user_id = %s
        """, (user_id, user_id, user_id))
        return [r[0] for r in cur.fetchall()]


def user_can_access_resolution(conn, user_id: int, resolution_id: int) -> bool:
    if _is_admin_or_super(conn, user_id):
        return True
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM resolution_management_resolution
            WHERE id = %s AND (resolution_owner_id = %s OR created_by_id = %s)
            LIMIT 1
        """, (resolution_id, user_id, user_id))
        if cur.fetchone():
            return True
        cur.execute("""
            SELECT 1 FROM resolution_management_resolutionsupportingteam
            WHERE resolution_id = %s AND user_id = %s
            LIMIT 1
        """, (resolution_id, user_id))
        return cur.fetchone() is not None


def get_user_access_flags(conn, user_id: int) -> Dict[str, bool]:
    """Get all access flags for a user in a single DB query."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT u.is_superuser,
                   ARRAY_AGG(DISTINCT rf.feature_id) FILTER (WHERE rf.feature_id IS NOT NULL) AS feature_ids
            FROM user_management_user u
            LEFT JOIN user_management_user_roles ur ON ur.user_id = u.id
            LEFT JOIN role_and_access_role_features rf ON rf.role_id = ur.role_id
            WHERE u.id = %s
            GROUP BY u.id, u.is_superuser
        """, (user_id,))
        row = cur.fetchone()

    if not row:
        return {k: False for k in ["superadmin", "all_projects", "budget",
                                    "user_management", "notes", "education", "project_docs",
                                    "sg_office_internal"]}

    sa = bool(row[0])
    feature_ids = set(row[1] or [])
    return {
        "superadmin": sa,
        "all_projects": sa or FeatureID.ALL_PROJECTS in feature_ids,
        "budget": sa or FeatureID.BUDGET_INFO in feature_ids,
        "user_management": sa or FeatureID.USER_MANAGEMENT in feature_ids,
        "notes": sa or FeatureID.NOTES in feature_ids,
        "education": sa or FeatureID.EDUCATION_DASH in feature_ids,
        "project_docs": sa or FeatureID.PROJECT_DOCS in feature_ids,
        "sg_office_internal": sa or FeatureID.SG_OFFICE_INTERNAL in feature_ids,
    }
