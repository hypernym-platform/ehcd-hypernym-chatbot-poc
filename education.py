"""
Education data helpers.
NOTE: FAISS-based search (build_tabular_documents, save_faiss, load_faiss,
search_tabular, update_tabular_index_if_changed) is DEPRECATED.
Education data now uses SQLite via edu_pg.py.
Still used: TabularConfig, _sync_blob_into_local, _sha256_file,
_load_hashes, _save_hashes, _read_excel_all_sheets, _normalize_df, _normalize_colname.
"""
from __future__ import annotations
import os, re, io, json, time, shutil, tempfile, hashlib, logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class TabularConfig:
    tabular_dir: str
    faiss_dir: str
    hash_json: str
    blob_conn_str: str = ""
    blob_container: str = ""
    blob_prefix: str = ""
    throttle_seconds: int = 600
    max_fields_per_doc: int = 40


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_hashes(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))
    return {}

def _save_hashes(path: str, data: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def _list_local_excels(tabular_dir: str) -> List[str]:
    if not os.path.exists(tabular_dir):
        return []
    return [
        os.path.join(tabular_dir, f)
        for f in os.listdir(tabular_dir)
        if f.lower().endswith((".xlsx", ".xls"))
    ]

def _sync_blob_into_local(cfg: TabularConfig) -> List[str]:
    """Download .xlsx/.xls files from Azure Blob if configured, else use local files."""
    if not (cfg.blob_conn_str and cfg.blob_container):
        logger.info("Blob env not set; using local Excel files.")
        return _list_local_excels(cfg.tabular_dir)

    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception:
        logger.warning("azure-storage-blob not installed; using local Excel files.")
        return _list_local_excels(cfg.tabular_dir)

    os.makedirs(cfg.tabular_dir, exist_ok=True)
    files = []
    try:
        svc = BlobServiceClient.from_connection_string(cfg.blob_conn_str)
        container = svc.get_container_client(cfg.blob_container)
        for blob in container.list_blobs(name_starts_with=cfg.blob_prefix):
            name = os.path.basename(blob.name)
            if not name.lower().endswith((".xlsx", ".xls")):
                continue
            local_path = os.path.join(cfg.tabular_dir, name)
            with open(local_path, "wb") as out:
                stream = container.download_blob(blob)
                out.write(stream.readall())
            files.append(local_path)
    except Exception as e:
        logger.error(f"Blob sync failed: {e}")
        return _list_local_excels(cfg.tabular_dir)
    return files

# ──────────────────────────────────────────────
# Schema & normalization
# ──────────────────────────────────────────────

def _normalize_colname(c: str) -> str:
    if not c: 
        return ""
    base = str(c).strip().lower()
    return (
        base.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("\n","_")
    )

def _read_excel_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        logger.error(f"Failed to open {path}: {e}")
        return {}
    frames: Dict[str, pd.DataFrame] = {}
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=s, dtype=object)
            frames[s] = df
        except Exception as e:
            logger.warning(f"Skip sheet {s} in {os.path.basename(path)}: {e}")
    return frames

# Human-friendly intro templates for known sheet name patterns
_SHEET_INTROS = {
    "determined_students": "### DETERMINED STUDENTS DATASET ###\nContains records of determined (special needs) students, including age groups, gender, disability types, support categories, school names, districts, provinces, and academic years.",
    "determined_schools": "### DETERMINED SCHOOLS DATASET ###\nContains information about schools supporting determined students, including school names, codes, sector, and region distribution.",
    "determined_staff": "### DETERMINED STAFF DATASET ###\nProvides details of staff assigned to support determined students, categorized by gender, nationality, sector, and year.",
    "staff_nationality": "### STAFF NATIONALITY DATASET ###\nShows the distribution of determined education staff by nationality, gender, and year.",
    "staff_gender": "### STAFF GENDER DATASET ###\nDetails about staff working in determined education, broken down by gender and year.",
    "students_nationality": "### DETERMINED STUDENTS NATIONALITY DATASET ###\nLists the number of determined students by nationality, gender, school, district, and year.",
    "students_total": "### TOTAL DETERMINED STUDENTS DATASET ###\nShows the total number of determined students by gender, grade, school, district, and year.",
    "students_disability": "### DETERMINED STUDENTS BY DISABILITY DATASET ###\nRecords of determined students categorized by disability type, age group, gender, school, district, and year.",
    "finance": "### EDUCATION FINANCE DATASET ###\nContains yearly financial information for education programs, including revenue and expenditure, broken down by district, province, and category.",
    "finance_overview": "### EDUCATION FINANCE OVERVIEW DATASET ###\nSummarized overview of annual revenue and expenditure for education by entity and program.",
    "general_education": "### GENERAL EDUCATION DATASET ###\nContains data on public and private schools, number of schools, students, teachers, and curricula across provinces and districts over the years.",
    "higher_education": "### HIGHER EDUCATION DATASET ###\nProvides information on higher education institutions, programs, degree levels, student enrollment, graduates, gender distribution, and nationalities.",
    "school_inspection": "### SCHOOL INSPECTION DATASET ###\nIncludes results of school inspections by year, school name, code, district, province, inspection scores, ratings, issues, and recommendations."
}

def _infer_dataset_kind(file_name: str, sheet_name: str) -> str:
    key = sheet_name.strip().lower().replace(" ","_")
    # Use sheet name as primary identifier
    return key

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    new_cols = {_c: _normalize_colname(_c) for _c in df.columns}
    df = df.rename(columns=new_cols)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).replace({"nan": None, "NaN": None}).str.strip()
    # NOTE: "year" is deliberately excluded — every education table uses it as a
    # plain 4-digit number (see EDU_SCHEMA_FOR_TOOL in edu_pg.py), never a calendar
    # date. Coercing it with pd.to_datetime turned 2022 into Timestamp('2022-01-01'),
    # which then got stored as the TEXT "2022-01-01 00:00:00" in SQLite — silently
    # breaking every LLM-generated "WHERE year = 2022" query (0 rows, no error).
    for dt in ["from_date","to_date","start_date","end_date","created_at","updated_at","date","birth_date","dob"]:
        if dt in df.columns:
            try:
                df[dt] = pd.to_datetime(df[dt], errors="coerce")
            except Exception:
                pass
    return df

def _row_to_text(row: pd.Series, sheet_key: str, cols: List[str], max_fields:int) -> str:
    """
    Convert row to human-readable text, tailored by sheet type (using sheet_key).
    """
    parts = []
    sk = sheet_key.lower()
    if sk == "determined_students":
        parts.append("### DETERMINED STUDENT RECORD ###")
        parts.append(f"Student {row.get('student_name','N/A')} (Gender: {row.get('gender','N/A')}, Age: {row.get('age','N/A')}) "
                     f"attends {row.get('school_name','N/A')} in {row.get('district','N/A')}, {row.get('province','N/A')}.")
        if row.get("disability"): parts.append(f"Disability: {row.get('disability','N/A')}, Support: {row.get('support_type','N/A')}, Status: {row.get('status','N/A')}.")
    elif sk == "determined_schools":
        parts.append("### DETERMINED SCHOOL RECORD ###")
        parts.append(f"School {row.get('school_name','N/A')} (Code: {row.get('school_code','N/A')}) "
                     f"is in {row.get('district','N/A')}, {row.get('province','N/A')} under sector {row.get('sector','N/A')}.")
    elif sk == "determined_staff":
        parts.append("### DETERMINED STAFF RECORD ###")
        parts.append(f"Staff {row.get('staff_name','N/A')} (Gender: {row.get('gender','N/A')}, Nationality: {row.get('nationality','N/A')}) "
                     f"is assigned to {row.get('school_name','N/A')} in {row.get('district','N/A')}, {row.get('province','N/A')} in year {row.get('year','N/A')}.")
    elif sk == "staff_nationality":
        parts.append("### STAFF NATIONALITY RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, in {row.get('district','N/A')}, {row.get('province','N/A')}, "
                     f"staff distribution by nationality: {row.to_dict()}.")
    elif sk == "staff_gender":
        parts.append("### STAFF GENDER RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, staff gender distribution: {row.to_dict()}.")
    elif sk == "students_nationality":
        parts.append("### STUDENTS NATIONALITY RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, school {row.get('school_name','N/A')} (district: {row.get('district','N/A')}, province: {row.get('province','N/A')}) "
                     f"had students by nationality and gender: {row.to_dict()}.")
    elif sk == "students_total":
        parts.append("### STUDENTS TOTAL RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, school {row.get('school_name','N/A')} (code: {row.get('school_code','N/A')}) "
                     f"in {row.get('district','N/A')}, {row.get('province','N/A')} had {row.get('enrollment_total','N/A')} students "
                     f"(male: {row.get('enrollment_male','N/A')}, female: {row.get('enrollment_female','N/A')}).")
    elif sk == "pass_fail_rate":
        parts.append("### PASS/FAIL RATE RECORD ###")
        parts.append(f"In {row.get('year','N/A')} grade {row.get('grade','N/A')} exam {row.get('exam_name','N/A')} "
                     f"subject {row.get('subject','N/A')} had {row.get('num_candidates','N/A')} candidates, "
                     f"{row.get('num_passed','N/A')} passed, {row.get('num_failed','N/A')} failed "
                     f"(Pass rate: {row.get('pass_rate','N/A')}%, Fail rate: {row.get('fail_rate','N/A')}%).")
    elif sk == "higher_education":
        parts.append("### HIGHER EDUCATION RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, {row.get('university','N/A')} (district: {row.get('district','N/A')}, {row.get('province','N/A')}) "
                     f"offered {row.get('degree_level','N/A')} in {row.get('department','N/A')} with total enrollment {row.get('enrollment_total','N/A')} "
                     f"(male: {row.get('enrollment_male','N/A')}, female: {row.get('enrollment_female','N/A')}), graduates {row.get('graduates','N/A')}.")
    elif sk == "test_scores":
        parts.append("### TEST SCORE RECORD ###")
        parts.append(f"In {row.get('year','N/A')}, grade {row.get('grade','N/A')} subject {row.get('subject','N/A')} "
                     f"exam {row.get('exam_name','N/A')} had average score {row.get('score_avg','N/A')}, "
                     f"median score {row.get('score_p50','N/A')}, top 10% score {row.get('score_p90','N/A')}, "
                     f"in {row.get('school_name','N/A')} ({row.get('district','N/A')}, {row.get('province','N/A')}).")
    else:
        parts.append(f"### DATA FROM SHEET: {sheet_key} ###")
        parts.append(str(row.to_dict()))

    return "\n".join(parts)


# ──────────────────────────────────────────────
# Document builder
# ──────────────────────────────────────────────

def build_tabular_documents(paths: List[str], cfg: TabularConfig) -> List[Document]:
    docs: List[Document] = []

    for path in paths:
        file_name = os.path.basename(path)
        frames = _read_excel_all_sheets(path)
        for sheet, raw in frames.items():
            df = _normalize_df(raw)
            if df.empty:
                continue

            sheet_key = sheet.strip().lower().replace(" ","_")
            intro_text = _SHEET_INTROS.get(sheet_key, f"### DATASET: {sheet} ###\nThis sheet contains data from {sheet}.")
            docs.append(Document(
                page_content=intro_text,
                metadata={
                    "section": "intro",
                    "file_name": file_name,
                    "sheet_name": sheet,
                }
            ))

            cols = list(df.columns)
            for i, row in df.fillna("").iterrows():
                row_text = _row_to_text(row, sheet_key, cols, cfg.max_fields_per_doc)
                md: Dict[str,Any] = {
                    "section":"row",
                    "file_name":file_name,
                    "sheet_name":sheet,
                    "row_index":int(i)
                }
                for c in cols:
                    v=row.get(c,"")
                    if v not in (None,"nan","NaN"):
                        md[c]=v
                docs.append(Document(page_content=row_text, metadata=md))

    return docs

# ──────────────────────────────────────────────
# FAISS Index Management
# ──────────────────────────────────────────────

def _atomic_replace_dir(src: str, dst: str):
    if os.path.exists(tmp := dst + ".tmp"):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(tmp, dst)

def save_faiss(docs: List[Document], embeddings, index_dir: str):
    if not docs:
        shutil.rmtree(index_dir, ignore_errors=True)
        return
    vs = FAISS.from_documents(docs, embeddings)
    tmp = tempfile.mkdtemp()
    vs.save_local(tmp)
    os.makedirs(os.path.dirname(index_dir) or ".", exist_ok=True)
    _atomic_replace_dir(tmp, index_dir)
    shutil.rmtree(tmp, ignore_errors=True)

def load_faiss(index_dir: str, embeddings) -> Optional[FAISS]:
    if not os.path.exists(index_dir):
        return None
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def scan_tabular_schema(cfg: TabularConfig) -> Dict[str, Dict[str, Any]]:
    paths = _sync_blob_into_local(cfg)
    report: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        file_name = os.path.basename(path)
        frames = _read_excel_all_sheets(path)
        report[file_name] = {}
        for sheet, raw in frames.items():
            df = _normalize_df(raw)
            report[file_name][sheet] = {
                "columns": list(df.columns),
                "sample_rows": df.head(2).to_dict(orient="records")
            }
    return report

def update_tabular_index_if_changed(cfg: TabularConfig, embeddings) -> bool:
    stamp_file = os.path.join(cfg.faiss_dir, ".last_check")
    now = int(time.time())
    try:
        if os.path.exists(stamp_file):
            last = int(open(stamp_file).read().strip() or "0")
            if (now - last) < cfg.throttle_seconds and os.path.exists(cfg.faiss_dir):
                return False
    except Exception:
        pass

    paths = _sync_blob_into_local(cfg)
    if not paths:
        logger.info("No Excel files found for Education tabular; skip.")
        return False

    prev = _load_hashes(cfg.hash_json)
    now_sha: Dict[str,str] = {}
    changed = False
    for p in paths:
        h=_sha256_file(p)
        now_sha[os.path.basename(p)] = h
        if prev.get(os.path.basename(p)) != h:
            changed=True

    if not changed and os.path.exists(cfg.faiss_dir):
        open(stamp_file,"w").write(str(now))
        logger.info("Education tabular index up-to-date.")
        return False

    df = pd.DataFrame()
    for p in paths:
        frames=_read_excel_all_sheets(p)
        for s, raw in frames.items():
            df = pd.concat(
                [
                    df,
                    _normalize_df(raw).assign(
                        _source_file=os.path.basename(p),
                        _source_sheet=s
                    ),
                ],
                ignore_index=True,
            )

    docs=build_tabular_documents(paths,cfg)
    save_faiss(docs,embeddings,cfg.faiss_dir)
    _save_hashes(cfg.hash_json,now_sha)
    open(stamp_file,"w").write(str(now))
    logger.info(f"Education tabular FAISS rebuilt: {len(docs)} docs from {len(paths)} files.")
    return True

def search_tabular(cfg: TabularConfig, embeddings, query:str, k:int=8,query_embedding: Optional[List[float]] = None)->List[Document]:
    vs=load_faiss(cfg.faiss_dir,embeddings)
    if not vs: return []
    try:
        if query_embedding is not None:
            return vs.similarity_search_by_vector(query_embedding, k=k)
        return vs.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"FAISS search error: {e}")
        return []
