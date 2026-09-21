"""
Education data manager — SQLite-backed.
Loads Excel files (from Azure Blob or local) into a local SQLite database.
GPT-4o generates SQL queries against these tables via tool calling.
"""

import os
import re
import json
import time
import sqlite3
import logging
import datetime
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

from education import (
    TabularConfig,
    _sync_blob_into_local,
    _sha256_file,
    _load_hashes,
    _save_hashes,
    _read_excel_all_sheets,
    _normalize_df,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite DB path
# ---------------------------------------------------------------------------
EDU_DB_PATH = os.getenv(
    "EDU_DB_PATH",
    os.path.join(os.getenv("DATA_ROOT", "./data"), "edu_data.db"),
)

# Thread-local connections for SQLite (thread-safety)
_local = threading.local()


def _get_edu_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "edu_conn") or _local.edu_conn is None:
        os.makedirs(os.path.dirname(EDU_DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(EDU_DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.edu_conn = conn
    return _local.edu_conn


# ---------------------------------------------------------------------------
# Sheet → Table mapping (only fact tables, skip DIM_* and trivial sheets)
# ---------------------------------------------------------------------------
SHEET_TABLE_MAP = {
    # GENERAL_EDUCATION.xlsx
    ("GENERAL_EDUCATION", "GENERAL_EDUCATION"): "edu_general_education",

    # HIGHER_EDUCATION.xlsx
    ("HIGHER_EDUCATION", "HIGHER_EDUCATION"): "edu_higher_education",
    ("HIGHER_EDUCATION", "INSTITUTIONS"): "edu_institutions",
    ("HIGHER_EDUCATION", "ENROLLMENT"): "edu_enrollment",
    ("HIGHER_EDUCATION", "FACULTY"): "edu_faculty",
    ("HIGHER_EDUCATION", "ADMINISTRATION"): "edu_administration",
    ("HIGHER_EDUCATION", "GRADUATES"): "edu_graduates",

    # TEST_SCORES.xlsx
    ("TEST_SCORES", "TEST_SCORES"): "edu_test_scores",
    ("TEST_SCORES", "PISA_READING"): "edu_pisa_reading",
    ("TEST_SCORES", "PISA_SCIENCE"): "edu_pisa_science",
    ("TEST_SCORES", "PISA_MATH"): "edu_pisa_math",
    ("TEST_SCORES", "TIMSS_SCIENCE"): "edu_timss_science",
    ("TEST_SCORES", "TIMSS_MATH"): "edu_timss_math",
    ("TEST_SCORES", "PIRLS_SCORE"): "edu_pirls_score",

    # PASS_FAIL_RATE.xlsx
    ("PASS_FAIL_RATE", "PASS_FAIL_RATE"): "edu_pass_fail_rate",

    # FINANCE_OVERVIEW.xlsx
    ("FINANCE_OVERVIEW", "FINANCE_OVERVIEW"): "edu_finance_overview",
    ("FINANCE_OVERVIEW", "TOTAL_REVENUE"): "edu_total_revenue",
    ("FINANCE_OVERVIEW", "TOTAL_EXPENDITURE"): "edu_total_expenditure",

    # DETERMINED_STUDENTS.xlsx
    ("DETERMINED_STUDENTS", "DETERMINED_STUDENTS"): "edu_determined_students",
    ("DETERMINED_STUDENTS", "DETERMINED_STAFF"): "edu_determined_staff",
    ("DETERMINED_STUDENTS", "DETERMINED_SCHOOLS"): "edu_determined_schools",
    ("DETERMINED_STUDENTS", "STAFF_NATIONALITY"): "edu_staff_nationality",
    ("DETERMINED_STUDENTS", "STAFF_GENDER"): "edu_staff_gender",
    ("DETERMINED_STUDENTS", "STUDENTS_NATIONALITY"): "edu_students_nationality",
    ("DETERMINED_STUDENTS", "STUDENTS_TOTAL"): "edu_students_total",
    ("DETERMINED_STUDENTS", "STUDENTS_DISABILITY"): "edu_students_disability",

    # SCHOOL_INSPECTION.xlsx
    ("SCHOOL_INSPECTION", "SCHOOL_INSPECTION"): "edu_school_inspection",

    # LABOUR.xlsx
    ("LABOUR", "Unemployment Rate For Populatio"): "edu_unemployment_rate",
    ("LABOUR", "Productive Families By Type Of "): "edu_productive_families_by_type",
    ("LABOUR", "Productive Families"): "edu_productive_families",
    ("LABOUR", "Percentage Distribution - Outsi"): "edu_percentage_distribution_outside",
    ("LABOUR", "Labour Force Participation Rate"): "edu_labour_force_participation",
}


# ---------------------------------------------------------------------------
# Schema description for GPT-4o tool (so it can generate correct SQL)
# ---------------------------------------------------------------------------
EDU_SCHEMA_FOR_TOOL = """
=== EDUCATION DATABASE SCHEMA (SQLite) ===
All table names start with 'edu_'. Generate valid SQLite SELECT queries only.
Use LIKE for case-insensitive text matching. Year columns are numeric.

TABLE: edu_general_education (3914 rows)
  year, education_type, region, gender, emirati_status, cycle, grade, curriculum,
  total_schools (numeric), total_students (numeric), total_staff (numeric), total_teachers (numeric),
  education_type_ar, region_ar, gender_ar, emirati_status_ar, cycle_ar, grade_ar
  -- education_type: 'Public', 'Private'
  -- region: 'Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Umm Al Quwain', 'Ras Al Khaimah', 'Fujairah'
  -- gender: 'Male', 'Female', 'Total'
  -- emirati_status: 'Emirati', 'Non-Emirati', 'Total'

TABLE: edu_higher_education (3127 rows)
  year, institution_name, region, sector, nationality, academic_degree,
  total_graduates (numeric), profession, total_faculty (numeric), season, gender,
  total_enrolled (numeric), region_ar, sector_ar, nationality_ar, gender_ar, season_ar
  -- sector: 'Federal', 'Private', 'Local'
  -- academic_degree: 'Bachelor', 'Master', 'PhD', 'Diploma'

TABLE: edu_enrollment (1312 rows)
  year, season, institution_name, gender, nationality, total_enrolled (numeric)

TABLE: edu_graduates (1597 rows)
  year, institution_name, region, sector, nationality, academic_degree, total_graduates (numeric)

TABLE: edu_faculty (73 rows)
  year, institution_name, region, sector, nationality, profession, total_faculty (numeric)

TABLE: edu_institutions (73 rows)
  institution_name, region, gender, sector

TABLE: edu_test_scores (8274 rows)
  region, school_name, school_type, curriculum, reading_score (numeric), year,
  test_name, math_score (numeric), grade, science_score (numeric), test_type,
  region_ar, school_type_ar, curriculum_ar, test_type_ar
  -- test_name: 'PISA', 'TIMSS', 'PIRLS'
  -- school_type: 'Government', 'Private'

TABLE: edu_pisa_reading, edu_pisa_science, edu_pisa_math (1040 rows each)
  region, school_name, school_type, curriculum, [reading/science/math]_score (numeric), year, test_name

TABLE: edu_timss_science, edu_timss_math (2343 rows each)
  region, school_name, curriculum, school_type, [science/math]_score (numeric), grade, year, test_name

TABLE: edu_pirls_score (468 rows)
  region, school_name, school_type, curriculum, reading_score (numeric), year, test_name

TABLE: edu_pass_fail_rate (2597 rows)
  year, education_type, region, cycle, grade, gender, emirati_status,
  pass_percentage (numeric), total_students (numeric),
  education_type_ar, region_ar, cycle_ar, grade_ar, gender_ar, emirati_status2

TABLE: edu_finance_overview (1896 rows)
  year, entity_name, description, revenue_budget (numeric), expense_budget (numeric),
  abbreviation, entity_name_ar, description_ar

TABLE: edu_total_revenue (330 rows)
  year, entity_name, description, revenue_budget (numeric)

TABLE: edu_total_expenditure (1566 rows)
  year, entity_name, description, expense_budget (numeric)

TABLE: edu_determined_students (152 rows)
  gender, disabiilty_type, age_group, 2021 (numeric), 2022 (numeric), 2023 (numeric),
  region, nationality, gender_ar, disability_type_ar, age_group_ar, region_ar, nationality_ar
  -- Note: year columns are named '2021', '2022', '2023' (quote them: "2021")

TABLE: edu_determined_schools (40 rows)
  sector, region, 2021 (numeric), 2022 (numeric), 2023 (numeric), sector_ar, region_ar

TABLE: edu_determined_staff (32 rows)
  sector, gender, year, total_staff (numeric), nationality, sector_ar, gender_ar, nationality_ar

TABLE: edu_staff_nationality (16 rows)
  sector, nationality, year, total_staff (numeric)

TABLE: edu_staff_gender (16 rows)
  sector, gender, year, total_staff (numeric)

TABLE: edu_students_nationality (16 rows)
  nationality, region, 2021 (numeric), 2022 (numeric), 2023 (numeric)

TABLE: edu_students_total (16 rows)
  gender, region, 2021 (numeric), 2022 (numeric), 2023 (numeric)

TABLE: edu_students_disability (120 rows)
  gender, disabiilty_type, age_group, 2021 (numeric), 2022 (numeric), 2023 (numeric)

TABLE: edu_school_inspection (395 rows)
  school_name, inspection_year, performance, performance_ar
  -- performance: 'Outstanding', 'Good', 'Acceptable', 'Weak'

TABLE: edu_unemployment_rate (24 rows)
  gender, education_level, 2019 (numeric), 2020 (numeric), 2023 (numeric), education_level2

TABLE: edu_productive_families_by_type (8 rows)
  production_type, 2022 (numeric), 2023 (numeric), production_type_ar

TABLE: edu_productive_families (18 rows)
  education_level, gender, 2022 (numeric), 2023 (numeric), education_level_ar, gender_ar

TABLE: edu_percentage_distribution_outside (22 rows)
  gender, reason, 2021 (numeric), 2022 (numeric), 2023 (numeric), gender_ar, reason_ar

TABLE: edu_labour_force_participation (26 rows)
  gender, education_level, 2019 (numeric), 2020 (numeric), 2023 (numeric), gender_ar, education_level_ar

IMPORTANT NOTES:
- Use LIKE (not ILIKE — SQLite is case-insensitive for LIKE by default for ASCII)
- For columns named as years ('2021', '2022', '2023'), quote them: SELECT "2021", "2022" FROM ...
- Use SUM(), AVG(), COUNT() for aggregations
- Always include LIMIT (max 200 rows)
- For Arabic queries, search both English and Arabic columns
- Do NOT use ILIKE — use LIKE instead (SQLite)
"""


# ---------------------------------------------------------------------------
# Load Excel → SQLite
# ---------------------------------------------------------------------------

def _sqlite_safe(v):
    """Convert a value to a type SQLite can bind (str, int, float, bytes, None)."""
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(v)
    return v


def _normalize_col_for_sql(col: str) -> str:
    """Normalize column name for SQL (lowercase, underscores)."""
    if not col:
        return "unnamed"
    s = str(col).strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    # If column starts with digit, keep as-is (will be quoted in SQL)
    return s or "unnamed"


def load_excel_to_sqlite(cfg: TabularConfig) -> bool:
    """
    Sync Excel files from Blob/local → load into SQLite.
    Returns True if any data was reloaded, False if up-to-date.
    """
    stamp_file = os.path.join(cfg.faiss_dir, ".edu_pg_stamp")
    os.makedirs(os.path.dirname(stamp_file) or ".", exist_ok=True)
    now = int(time.time())

    # Throttle check
    try:
        if os.path.exists(stamp_file):
            with open(stamp_file) as f:
                last = int(f.read().strip() or "0")
            if (now - last) < cfg.throttle_seconds and os.path.exists(EDU_DB_PATH):
                return False
    except Exception:
        pass

    # Sync from Blob
    paths = _sync_blob_into_local(cfg)
    if not paths:
        logger.info("[EduPG] No Excel files found; skip.")
        return False

    # Hash check — only reload if files changed
    prev = _load_hashes(cfg.hash_json)
    now_sha: Dict[str, str] = {}
    changed = False
    for p in paths:
        h = _sha256_file(p)
        now_sha[os.path.basename(p)] = h
        if prev.get(os.path.basename(p)) != h:
            changed = True

    if not changed and os.path.exists(EDU_DB_PATH):
        with open(stamp_file, "w") as f:
            f.write(str(now))
        logger.info("[EduPG] Education SQLite up-to-date.")
        return False

    # Load all sheets into SQLite
    conn = sqlite3.connect(EDU_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    tables_loaded = 0

    try:
        for path in paths:
            file_stem = os.path.splitext(os.path.basename(path))[0]
            frames = _read_excel_all_sheets(path)

            for sheet_name, raw_df in frames.items():
                table_key = (file_stem, sheet_name)
                table_name = SHEET_TABLE_MAP.get(table_key)
                if not table_name:
                    continue  # skip unmapped sheets (DIM_*, Sheet1, etc.)

                df = _normalize_df(raw_df)
                if df.empty:
                    continue

                # Normalize column names
                col_map = {}
                for c in df.columns:
                    normalized = _normalize_col_for_sql(c)
                    # Handle duplicates
                    if normalized in col_map.values():
                        i = 2
                        while f"{normalized}_{i}" in col_map.values():
                            i += 1
                        normalized = f"{normalized}_{i}"
                    col_map[c] = normalized
                df = df.rename(columns=col_map)

                # Replace NaN with None
                df = df.where(pd.notnull(df), None)
                # Also replace string "nan"
                df = df.replace({"nan": None, "NaN": None, "None": None})

                # Drop and recreate table
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                cols_sql = ", ".join(f'"{c}" TEXT' for c in df.columns)
                conn.execute(f'CREATE TABLE "{table_name}" ({cols_sql})')

                # Bulk insert
                if not df.empty:
                    placeholders = ", ".join(["?"] * len(df.columns))
                    insert_sql = f'INSERT INTO "{table_name}" ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in df.columns)}) VALUES ({placeholders})'
                    rows = [tuple(_sqlite_safe(v) for v in row) for row in df.itertuples(index=False, name=None)]
                    conn.executemany(insert_sql, rows)

                tables_loaded += 1
                logger.info(f"[EduPG] Loaded {table_name}: {len(df)} rows, {len(df.columns)} cols")

        conn.commit()
    except Exception as e:
        logger.error(f"[EduPG] Failed to load tables: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    # Update hashes and stamp
    _save_hashes(cfg.hash_json, now_sha)
    with open(stamp_file, "w") as f:
        f.write(str(now))
    logger.info(f"[EduPG] Education SQLite loaded: {tables_loaded} tables from {len(paths)} files.")

    # Reset thread-local connections so they pick up the new data
    if hasattr(_local, "edu_conn") and _local.edu_conn:
        try:
            _local.edu_conn.close()
        except Exception:
            pass
        _local.edu_conn = None

    return True


# ---------------------------------------------------------------------------
# Safe SQL executor
# ---------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)
_REQUIRES_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def execute_education_sql(sql_query: str, row_limit: int = 200) -> dict:
    """
    Validate and execute a SELECT-only query against edu_* tables in SQLite.
    Returns {"columns": [...], "rows": [...], "row_count": n}
    or {"error": "..."}
    """
    # 1. Must start with SELECT
    if not _REQUIRES_SELECT.match(sql_query):
        return {"error": "Only SELECT queries are permitted."}

    # 2. Must not contain any DML/DDL
    if _FORBIDDEN.search(sql_query):
        return {"error": "Query contains forbidden statements."}

    # 3. Must only reference edu_* tables
    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+\"?([a-zA-Z_][a-zA-Z0-9_]*)\"?", sql_query, re.IGNORECASE
    )
    for tref in table_refs:
        if not tref.lower().startswith("edu_"):
            return {"error": f"Query references non-education table: {tref}. Only edu_* tables are allowed."}

    # 4. Inject LIMIT if not present
    if not re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE):
        sql_query = sql_query.rstrip("; \n") + f" LIMIT {row_limit}"

    # 5. Execute
    try:
        conn = _get_edu_conn()
        cur = conn.cursor()
        cur.execute(sql_query)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchmany(row_limit)
        return {
            "columns": cols,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except sqlite3.OperationalError as e:
        err_msg = str(e)
        if "no such table" in err_msg:
            return {"error": "Education data is not yet available. Tables are being initialized — please retry in a moment."}
        return {"error": f"Query failed: {err_msg}"}
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}
