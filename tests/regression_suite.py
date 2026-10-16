"""
Permanent regression suite for the EHCD chatbot's routing/chart pipeline.

Runs LIVE against the real GPT-4o router and the real dev Postgres DB — not
mocks — matching how every bug in this pipeline has actually been found and
fixed. Ground-truth expectations for data-dependent cases are computed by
querying the DB directly at run time, not hardcoded, so the suite stays
valid as the dev dataset changes.

Run after every change to app.py / tools.py / chart_engine.py / db_queries.py
/ edu_pg.py.

Usage:
    python tests/regression_suite.py                      # run everything
    python tests/regression_suite.py --category chart_intent
    python tests/regression_suite.py --list-categories
    python tests/regression_suite.py -v                    # print full responses

Scored BY CATEGORY, not just one aggregate pass/fail — a single number can
hide a 100% failure rate in one category behind a healthy overall score.
"""
import argparse
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import logging
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

import app
import tools as tools_mod
from tools import (
    run_chatbot_graph, build_available_tools, detect_unsupported_action,
    answer_node,
)
from chart_engine import (
    detect_chart_opportunity, detect_palette_from_query, recent_turn_had_chart,
    last_user_query, CHART_SHOWN_MARKER, _encode_palette_marker,
    validate_output_consistency,
)
from db_queries import list_projects, list_sg_office_meetings
from edu_pg import execute_education_sql

VERBOSE = False

# Real users in the dev DB (see conversation history for how these were
# identified — superadmin is the general test identity used throughout
# this project; Marwan is a genuinely RBAC-restricted PM with real
# projects, used for the access-control tests).
SUPERADMIN_ID = 103
SUPERADMIN_NAME = "Super Admin"
MARWAN_ID = 80
MARWAN_NAME = "Marwan Ahmed Mohamed"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self):
        with app.pg_conn() as conn:
            self.superadmin_tools = build_available_tools(conn, SUPERADMIN_ID)
            self.marwan_tools = build_available_tools(conn, MARWAN_ID)

    def run_turn(self, query, history, user_id=SUPERADMIN_ID, user_name=SUPERADMIN_NAME, tools=None):
        """Runs one live turn through the real graph, mirroring app.py's own
        pipeline (router -> tools -> answer -> chart detection) exactly, so
        a pass here means the real app would behave the same way."""
        tools = tools or self.superadmin_tools
        recent_had_chart = recent_turn_had_chart(history)
        previous_query = last_user_query(history)
        history.append({"role": "user", "content": query})
        tool_results = []
        chart_intent = {}
        resp = ""
        for chunk in run_chatbot_graph(
            query=query, conversation_history=history, available_tools=tools,
            user_id=user_id, user_name=user_name, user_role="", user_email="", user_contact_no="",
            client=app.client, model=app.cfg.AZURE_OPENAI_DEPLOYMENT, pg_conn_fn=app.pg_conn,
            policy_cfg=app.POLICY_CFG, emb=app.emb, tool_results_collector=tool_results,
            chart_intent_collector=chart_intent,
        ):
            resp += chunk
        palette = detect_palette_from_query(query, history)
        chart_data = None
        try:
            chart_data = detect_chart_opportunity(
                query, tool_results, resp, palette=palette,
                recent_had_chart=recent_had_chart, previous_query=previous_query,
                want_chart=chart_intent.get("want_chart"), router_chart_type=chart_intent.get("chart_type"),
            )
        except Exception:
            pass
        history_text = app._strip_html_for_history(resp)
        if chart_data:
            history_text += CHART_SHOWN_MARKER + _encode_palette_marker(palette)
        history.append({"role": "assistant", "content": history_text})
        if VERBOSE:
            safe = resp[:300].encode("ascii", "replace").decode("ascii")
            print(f"      [turn] Q={query!r}")
            print(f"      want_chart={chart_intent.get('want_chart')} chart_type={chart_intent.get('chart_type')} rows={len(tool_results)}")
            print(f"      resp={safe}")
        return TurnResult(resp, tool_results, chart_intent, chart_data, history)


class TurnResult:
    def __init__(self, resp, tool_results, chart_intent, chart_data, history):
        self.resp = resp
        self.tool_results = tool_results
        self.chart_intent = chart_intent
        self.chart_data = chart_data
        self.history = history

    def chart_trace(self, index=0):
        """Reads the REAL top-level shape detect_chart_opportunity returns —
        chart_data["plotly_data"][i] has x/y (bar/line) or labels/values
        (pie), plus "name" for the series label. Returns (labels, values, name)."""
        if not self.chart_data:
            return None
        traces = self.chart_data.get("plotly_data") or []
        if index >= len(traces):
            return None
        trace = traces[index]
        labels = trace.get("x") or trace.get("labels") or []
        values = trace.get("y") or trace.get("values") or []
        return labels, values, trace.get("name")


def ground_truth_project_count(conn, name_substr):
    """Independent re-derivation of 'projects X is included in' (manager OR
    team member) directly in SQL, for comparing the bot's own answer
    against — NOT a reuse of _multi_word_ilike, so a bug shared between the
    app's query and this check wouldn't silently cancel out. Still ANDs
    each word separately rather than one contiguous substring, though — a
    plain '%abdalla alzaabi%' ILIKE never matches the real stored name
    "Abdalla Khalid Alzaabi" (middle name breaks contiguous matching),
    which would make this ground truth wrong, not the app, for every name
    with a middle name — the exact bug _multi_word_ilike was already built
    to avoid in the app; this check needs the same correct semantics via
    its own independent code path, not a resurrection of that old bug."""
    words = name_substr.split()
    mgr_clause = " AND ".join(["u.full_name_en ILIKE %s"] * len(words))
    tm_clause = " AND ".join(["name_en ILIKE %s"] * len(words))
    params = [f"%{w}%" for w in words] + [f"%{w}%" for w in words]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(DISTINCT p.id) FROM project_management_project p
            LEFT JOIN user_management_user u ON u.id = p.project_manager_id
            WHERE ({mgr_clause})
               OR p.id IN (SELECT project_id FROM project_management_teammember WHERE {tm_clause})
            """,
            params,
        )
        return cur.fetchone()[0]


def ground_truth_total_projects(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM project_management_project")
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

REGISTRY = []  # list of (category, name, fn)


def test(category):
    def deco(fn):
        REGISTRY.append((category, fn.__name__, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# Category: unsupported_action — pure function, no LLM
# ---------------------------------------------------------------------------

@test("unsupported_action")
def test_unsupported_action_detects_writes(h):
    must_flag = [
        "change the Chatbot project status to ongoing",
        "update the project status to completed",
        "please update the project status to completed",
        "set the priority to high for this task",
        "approve this resolution",
        "delete the meeting",
        "assign Rashid to the Chatbot project",
        "create a new project called Chatbot",
    ]
    for q in must_flag:
        assert detect_unsupported_action(q), f"should have been flagged as unsupported: {q!r}"


@test("unsupported_action")
def test_unsupported_action_allows_reads_and_charts(h):
    must_not_flag = [
        "change the colors",
        "create me a graph about projects",
        "list all projects",
        "what is the status of the Chatbot project",
        "show me delayed projects",
        "give chart of projects vs budgets",
        "which tasks were created this month",
        "can you update me on the project status",
        "is this meeting cancelled or not",
        "how many projects does Abdalla Khalid manage",
        "bar chart of projects vs status",
        "make it orange",
    ]
    for q in must_not_flag:
        assert not detect_unsupported_action(q), f"false positive — should NOT have been flagged: {q!r}"


# ---------------------------------------------------------------------------
# Category: palette — pure function, no LLM
# ---------------------------------------------------------------------------

@test("palette")
def test_palette_rotates_without_repeating(h):
    from chart_engine import detect_palette_from_query, _encode_palette_marker
    history = []
    first = detect_palette_from_query("create me a graph about projects", history)
    assert first == "brown", f"fresh chart should default to brown palette, got {first!r}"
    history.append({"role": "user", "content": "create me a graph about projects"})
    history.append({"role": "assistant", "content": "chart shown" + CHART_SHOWN_MARKER + _encode_palette_marker(first)})

    seen = [first]
    for _ in range(4):
        p = detect_palette_from_query("change the colors", history)
        assert p != seen[-1], f"palette repeated consecutively: {seen + [p]}"
        seen.append(p)
        history.append({"role": "user", "content": "change the colors"})
        history.append({"role": "assistant", "content": "chart shown" + CHART_SHOWN_MARKER + _encode_palette_marker(p)})
    assert len(set(seen)) >= 4, f"palette rotation too repetitive: {seen}"


@test("palette")
def test_palette_explicit_color_wins(h):
    from chart_engine import detect_palette_from_query
    history = [
        {"role": "user", "content": "bar chart of projects vs status"},
        {"role": "assistant", "content": "chart shown" + CHART_SHOWN_MARKER},
    ]
    p = detect_palette_from_query("make it blue", history)
    assert p == "blue", f"explicit color request should win outright, got {p!r}"


# ---------------------------------------------------------------------------
# Category: output_validation — pure function, no LLM
# ---------------------------------------------------------------------------

@test("output_validation")
def test_output_validation_catches_duplicate_counting(h):
    # Real top-level shape: chart_data["plotly_data"] is a list of Plotly
    # traces (x/y + name), not the internal {"labels":[...],"datasets":[...]}
    # shape some chart_engine helpers use before converting to this.
    chart_doubled = {"plotly_data": [{"x": ["Rashid", "Abdalla Alzaabi"], "y": [10, 6], "name": "Count", "type": "bar"}]}
    tool_results = [{"id": i} for i in range(8)]
    text = "Abdalla Alzaabi is involved in 3 projects, while Rashid is involved in 5 projects."
    warnings = validate_output_consistency(chart_doubled, tool_results, text)
    assert warnings, "doubled chart data (16 counted vs 8 real rows) should have produced warnings"
    assert any("EXCEEDS" in w for w in warnings), f"expected a row-count-exceeded warning, got: {warnings}"


@test("output_validation")
def test_output_validation_silent_on_correct_data(h):
    chart_correct = {"plotly_data": [{"x": ["Rashid", "Abdalla Alzaabi"], "y": [5, 3], "name": "Count", "type": "bar"}]}
    tool_results = [{"id": i} for i in range(8)]
    text = "Abdalla Alzaabi is involved in 3 projects, while Rashid is involved in 5 projects."
    warnings = validate_output_consistency(chart_correct, tool_results, text)
    assert not warnings, f"correct, consistent data should produce no warnings, got: {warnings}"


@test("output_validation")
def test_output_validation_ignores_non_count_datasets(h):
    chart_budget = {"plotly_data": [{"x": ["P1", "P2"], "y": [9999999, 5000000], "name": "Allocated", "type": "bar"}]}
    tool_results = [{"id": i} for i in range(2)]
    warnings = validate_output_consistency(chart_budget, tool_results, "Budgets shown below.")
    assert not warnings, f"budget-type datasets should never be checked against row count, got: {warnings}"


# ---------------------------------------------------------------------------
# Category: result_state — answer_node directly, no router needed
# ---------------------------------------------------------------------------

def _answer_with_state(had_technical_error, had_empty_result, query):
    import queue
    state = {
        "user_name": "Test User", "user_role": "", "user_email": "", "user_contact_no": "",
        "messages": [
            {"role": "system", "content": "placeholder"},
            {"role": "user", "content": query},
        ],
        "tool_results_for_chart": [],
        "want_chart": False,
        "had_technical_error": had_technical_error,
        "had_empty_result": had_empty_result,
        "chunk_queue": queue.Queue(),
        "client": app.client,
        "model": app.cfg.AZURE_OPENAI_DEPLOYMENT,
    }
    return answer_node(state).get("final_response", "")


@test("result_state")
def test_technical_error_phrased_distinctly(h):
    resp = _answer_with_state(True, False, "how many projects does Rashid manage")
    low = resp.lower()
    assert "no data" not in low and "does not" not in low and "0 " not in low, (
        f"a technical error must not be phrased as 'no data found' or a plain zero — got: {resp!r}"
    )
    assert any(w in low for w in ("technical", "problem", "wrong", "try again", "error")), (
        f"a technical error should clearly say something went wrong — got: {resp!r}"
    )


@test("result_state")
def test_empty_result_phrased_distinctly(h):
    resp = _answer_with_state(False, True, "how many projects does Rashid manage")
    low = resp.lower()
    assert not any(w in low for w in ("technical", "problem retrieving", "try again")), (
        f"a legitimate empty result must not be phrased as a technical failure — got: {resp!r}"
    )


# ---------------------------------------------------------------------------
# Category: date_filters — direct db_queries calls, no LLM needed
# ---------------------------------------------------------------------------

@test("date_filters")
def test_upcoming_meetings_excludes_past_dates(h):
    """Live bug: status='upcoming' meant Confirmed/Rescheduled ONLY, with
    no date check at all — a Confirmed meeting from weeks ago still showed
    up as 'upcoming'. Fixed by ANDing the date condition into the status
    branch in list_sg_office_meetings (see MEETING_UPCOMING_STATUSES)."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT (NOW() AT TIME ZONE 'Asia/Dubai')::date")
            today = cur.fetchone()[0]
        result = list_sg_office_meetings(conn, SUPERADMIN_ID, filters={"status": "upcoming"})
    assert "error" not in result, f"unexpected error: {result}"
    for row in result.get("data", []):
        d = row.get("scheduled_date")
        assert d is None or d >= today, (
            f"'upcoming' returned a meeting dated {d}, which is before today ({today}) — "
            f"the date check regressed. row={row.get('organization')!r}"
        )


@test("date_filters")
def test_named_month_filter_matches_ground_truth(h):
    """Direct check of the scheduled_month filter itself (see the live
    end-to-end version below for the router-level enforcement)."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest WHERE EXTRACT(MONTH FROM scheduled_date) = 10")
            expected = cur.fetchone()[0]
        result = list_sg_office_meetings(conn, SUPERADMIN_ID, filters={"scheduled_month": 10})
    assert result.get("total_count") == expected, f"expected {expected}, got {result.get('total_count')}"


@test("date_filters")
def test_generic_date_filter_rejects_unknown_column(h):
    """The (column, operator, value) escape hatch's safety property: a
    column name outside the whitelist must be silently ignored, never
    reach the query string — confirms there's no injection surface, not
    just that valid input works."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest")
            expected_total = cur.fetchone()[0]
        result = list_sg_office_meetings(conn, SUPERADMIN_ID, filters={
            "date_filters": [{"column": "status; DROP TABLE x; --", "op": "eq", "value": "2026-10-01"}]
        })
    assert result.get("total_count") == expected_total, (
        f"an unwhitelisted column should be silently ignored (all {expected_total} rows), "
        f"got {result.get('total_count')} — the filter may be reaching the query unvalidated"
    )


@test("date_filters")
def test_october_meetings_live_enforcement(h):
    """Live bug, reproduced 3/3 trials: "list meetings scheduled for
    October" dropped the date constraint entirely — the schema had no
    deterministic way to express a named month, only scheduled_after/
    scheduled_before, which need the model to compute exact day
    boundaries itself (it didn't even try). Fixed two ways: a new
    scheduled_month filter (removes the need for the model to compute
    anything), AND a deterministic enforcement check in
    tool_executor_node that parses the month from the query text directly
    and injects it if the tool call came back with no date constraint at
    all — so this holds regardless of what the router does on a given
    sample, not just usually."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest WHERE EXTRACT(MONTH FROM scheduled_date) = 10")
            expected = cur.fetchone()[0]
    r = h.run_turn("List all meetings scheduled for October with date, requester, venue and status.", [])
    assert len(r.tool_results) == expected, (
        f"expected {expected} October meetings, got {len(r.tool_results)} — "
        f"the date constraint was likely dropped again"
    )


@test("date_filters")
def test_all_meetings_in_month_keeps_date_not_status(h):
    """Edge case between the two deterministic guards: "all meetings in
    October" must keep the month scope (named month present) while still
    meaning "no status filter" — not collapse to either all 28 unfiltered
    or an empty result from the two guards fighting each other."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest WHERE EXTRACT(MONTH FROM scheduled_date) = 10")
            expected = cur.fetchone()[0]
    r = h.run_turn("Create a chart of all external meetings in October.", [])
    assert len(r.tool_results) == expected, (
        f"expected {expected} (October-scoped), got {len(r.tool_results)} — "
        f"the 'all X' and month-enforcement guards may be conflicting"
    )


# ---------------------------------------------------------------------------
# Category: rbac — direct db_queries calls, no LLM needed (fast, reliable)
# ---------------------------------------------------------------------------

@test("rbac")
def test_restricted_user_sees_own_projects(h):
    with app.pg_conn() as conn:
        result = list_projects(conn, MARWAN_ID, filters={"project_manager": "marwan"})
    assert "error" not in result, f"Marwan asking about his own projects should not be denied: {result}"
    assert result.get("total_count", 0) >= 1, f"Marwan should have at least 1 real project, got: {result}"


@test("rbac")
def test_restricted_user_denied_another_pms_projects(h):
    with app.pg_conn() as conn:
        result = list_projects(conn, MARWAN_ID, filters={"project_manager": "abdalla khalid"})
    assert "error" in result, f"Marwan asking about a different PM's projects should be denied, got: {result}"
    assert "access denied" in result["error"].lower(), f"expected an access-denied message, got: {result}"


@test("rbac")
def test_superadmin_not_denied(h):
    with app.pg_conn() as conn:
        result = list_projects(conn, SUPERADMIN_ID, filters={"project_manager": "abdalla khalid"})
    assert "error" not in result, f"superadmin should never be denied, got: {result}"


# ---------------------------------------------------------------------------
# Category: sql_fabrication_guard — direct edu_pg call, no LLM needed
# ---------------------------------------------------------------------------

@test("sql_fabrication_guard")
def test_from_less_literal_query_rejected(h):
    result = execute_education_sql(
        "SELECT 'Abdalla Alzaabi' AS name, 3 AS project_count UNION SELECT 'Rashid' AS name, 5 AS project_count"
    )
    assert "error" in result, f"a FROM-less literal-only query must be rejected, got: {result}"


@test("sql_fabrication_guard")
def test_legit_education_query_still_works(h):
    result = execute_education_sql("SELECT year, region FROM edu_general_education LIMIT 3")
    assert "error" not in result, f"a legitimate edu_* query should still work, got: {result}"
    assert result.get("row_count", 0) > 0, f"expected real rows back, got: {result}"


@test("sql_fabrication_guard")
def test_non_edu_table_rejected(h):
    result = execute_education_sql("SELECT * FROM project_management_project LIMIT 1")
    assert "error" in result, f"a query against a non-edu_* table must be rejected, got: {result}"


# ---------------------------------------------------------------------------
# Category: chart_intent — live
# ---------------------------------------------------------------------------

@test("chart_intent")
def test_axis_description_triggers_chart_without_chart_word(h):
    r = h.run_turn(
        "based on projects, i want to see the number of projects abdalla Khalid is included in. "
        "Xaxis should represent name and yaxis should represent number of projects",
        [],
    )
    assert r.chart_intent.get("want_chart") is True, f"axis-description phrasing should trigger want_chart=True, got {r.chart_intent}"
    assert r.chart_data is not None, "expected a chart to be built"


@test("chart_intent")
def test_plain_list_query_no_chart(h):
    r = h.run_turn("list all tasks", [])
    assert r.chart_intent.get("want_chart") is not True, f"a plain list query should not want a chart, got {r.chart_intent}"
    assert r.chart_data is None, "no chart should have been built for a plain list query"


@test("chart_intent")
def test_plain_count_question_no_chart(h):
    r = h.run_turn("how many projects does Abdalla Khalid manage", [])
    assert r.chart_intent.get("want_chart") is not True, (
        f"a plain factual question (not asking for a visual) should not want a chart, got {r.chart_intent}"
    )


# ---------------------------------------------------------------------------
# Category: chart_type — live, tests type override persistence
# ---------------------------------------------------------------------------

@test("chart_type")
def test_explicit_type_override_wins_after_prior_chart(h):
    history = []
    r1 = h.run_turn("line chart of tasks by status", history)
    assert r1.chart_intent.get("chart_type") == "line", f"expected line chart_type, got {r1.chart_intent}"
    history = r1.history
    r2 = h.run_turn("now show it as a bar chart", history)
    assert r2.chart_intent.get("chart_type") == "bar", f"explicit type override should win, got {r2.chart_intent}"
    labels1, data1, _ = r1.chart_trace()
    labels2, data2, _ = r2.chart_trace()
    assert data1 == data2, f"underlying data should be preserved across a pure type change: {data1} vs {data2}"


# ---------------------------------------------------------------------------
# Category: subject_count_chart — live, ground-truth verified
# ---------------------------------------------------------------------------

@test("subject_count_chart")
def test_single_subject_count_matches_ground_truth(h):
    with app.pg_conn() as conn:
        expected = ground_truth_project_count(conn, "abdalla khalid")
    r = h.run_turn(
        "based on projects, i want to see the number of projects abdalla Khalid is included in. "
        "Xaxis should represent name and yaxis should represent number of projects",
        [],
    )
    assert r.chart_data, "expected a chart"
    labels, data, dataset_label = r.chart_trace()
    assert dataset_label == "Count", f"expected a 'Count' dataset for a single-subject chart, got {dataset_label!r}"
    assert len(data) == 1, f"expected exactly one bar for one subject, got {data}"
    assert data[0] == expected, f"chart shows {data[0]} but ground truth (direct SQL) is {expected}"
    assert not validate_output_consistency(r.chart_data, r.tool_results, r.resp), (
        "output-consistency validator should be silent on a correct single-subject chart"
    )


@test("subject_count_chart")
def test_multi_subject_count_matches_ground_truth(h):
    with app.pg_conn() as conn:
        expected_alzaabi = ground_truth_project_count(conn, "abdalla alzaabi")
        expected_rashid = ground_truth_project_count(conn, "rashid")
    r = h.run_turn(
        "based on the projects i want to see the number of projects abdalla Alzaabi and rashid are included in. "
        "I want the x axis to represent their name, while the y axis represent the number of projects they are included in",
        [],
    )
    assert r.chart_data, "expected a chart"
    labels, data, dataset_label = r.chart_trace()
    assert dataset_label == "Count", f"expected a 'Count' dataset, got {dataset_label!r}"
    assert len(labels) == 2, f"expected two bars (one per subject), got labels={labels}"
    by_label = dict(zip([l.lower() for l in labels], data))
    alzaabi_val = next((v for k, v in by_label.items() if "alzaabi" in k), None)
    rashid_val = next((v for k, v in by_label.items() if "rashid" in k), None)
    assert alzaabi_val == expected_alzaabi, f"Alzaabi's bar shows {alzaabi_val}, ground truth is {expected_alzaabi}"
    assert rashid_val == expected_rashid, f"Rashid's bar shows {rashid_val}, ground truth is {expected_rashid}"
    assert not validate_output_consistency(r.chart_data, r.tool_results, r.resp), (
        "output-consistency validator should be silent on a correct multi-subject chart — "
        "a non-empty result here means the duplicate-tool-call doubling bug has regressed"
    )


# ---------------------------------------------------------------------------
# Category: pm_matching — live
# ---------------------------------------------------------------------------

@test("pm_matching")
def test_half_name_matches_full_name(h):
    with app.pg_conn() as conn:
        expected = ground_truth_project_count(conn, "alzaabi")
    assert expected > 0, "test precondition failed: no projects found for 'alzaabi' at all"
    r = h.run_turn("how many projects is Alzaabi included in", [])
    # Loose check: the response should mention a number consistent with
    # having found real data, not an empty/zero/denied answer — ambiguous
    # half-name matches (multiple Alzaabis exist) make an exact count
    # assertion unreliable, so this checks the mechanism didn't come up
    # empty rather than pinning an exact number.
    assert len(r.tool_results) > 0, f"half-name 'Alzaabi' alone should have matched real rows, got 0. resp={r.resp!r}"


# ---------------------------------------------------------------------------
# Category: context_reset — live, ground-truth verified
# ---------------------------------------------------------------------------

@test("context_reset")
def test_list_all_projects_resets_after_filtered_queries(h):
    with app.pg_conn() as conn:
        expected_total = ground_truth_total_projects(conn)
    history = []
    r1 = h.run_turn(
        "based on the projects i want to see the number of projects abdalla Alzaabi and rashid are included in. "
        "I want the x axis to represent their name, while the y axis represent the number of projects they are included in",
        history,
    )
    history = r1.history
    r2 = h.run_turn("list all projects", history)
    assert len(r2.tool_results) == expected_total, (
        f"'list all projects' after two filtered turns should fetch all {expected_total} projects fresh, "
        f"got {len(r2.tool_results)} — likely stale/contaminated context"
    )
    assert r2.chart_data is None, "a plain 'list all projects' should not produce a chart"
    assert "alzaabi" not in r2.resp.lower() and "rashid" not in r2.resp.lower(), (
        f"response should not bleed in names from the prior filtered turns: {r2.resp!r}"
    )


@test("context_reset")
def test_all_meetings_resets_after_upcoming_filter(h):
    """Live bug, reproduced 2/3 times even with a router prompt rule added
    against it — fixed with a deterministic post-hoc strip in
    tool_executor_node (see _BROADENING_RE / _MEETING_STATUS_WORDS_RE),
    since the prompt rule alone wasn't reliable. This test is what would
    catch that guard breaking again."""
    with app.pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest")
            expected_total = cur.fetchone()[0]
    history = []
    r1 = h.run_turn("list all upcoming external meetings", history)
    history = r1.history
    r2 = h.run_turn("Create a chart of all external meetings", history)
    assert len(r2.tool_results) == expected_total, (
        f"'all external meetings' after an 'upcoming' turn should fetch all {expected_total} meetings "
        f"fresh, got {len(r2.tool_results)} — the status='upcoming' filter likely leaked from the prior turn"
    )
    assert "upcoming" not in r2.resp.lower(), (
        f"response should not describe this as 'upcoming' — that word was never in this turn's query: {r2.resp!r}"
    )


# ---------------------------------------------------------------------------
# Category: readiness_charts / budget_charts — live, structural checks
# ---------------------------------------------------------------------------

@test("readiness_charts")
def test_combined_readiness_chart_has_two_series(h):
    r = h.run_turn("generate a chart of meetings vs readiness status", [])
    assert r.chart_data, "expected a chart"
    names = {t.get("name") for t in r.chart_data.get("plotly_data", [])}
    assert "Facility Readiness" in names and "Visitor Readiness" in names, (
        f"expected both Facility and Visitor Readiness series, got: {names}"
    )


@test("budget_charts")
def test_budget_chart_has_three_series(h):
    r = h.run_turn("show me a chart of project budgets", [])
    assert r.chart_data, "expected a chart"
    names = {t.get("name") for t in r.chart_data.get("plotly_data", [])}
    assert {"Allocated", "Spent", "Remaining"} <= names, f"expected Allocated/Spent/Remaining, got: {names}"


# ---------------------------------------------------------------------------
# Category: sg_office_charts — live, three real bugs found and fixed in
# one batch (Email/Memo/Weekly Action modules)
# ---------------------------------------------------------------------------

@test("sg_office_charts")
def test_weekly_actions_entity_name_does_not_trigger_date_bucketing(h):
    """Live bug: "Weekly Actions" (this app's own module name) contains
    the literal word "weekly", which false-triggered the date-bucket
    chart case — "Weekly Actions by priority" got charted by date instead
    of by priority, with the text describing a priority breakdown that
    didn't match the actual chart at all. Fixed with a negative lookahead
    (weekly(?!\\s+actions?)) in chart_engine._DATE_BUCKET_RE."""
    r = h.run_turn("Create a pie chart showing Weekly Actions by priority: Low, Medium and High.", [])
    assert r.chart_data, "expected a chart"
    t0 = r.chart_data["plotly_data"][0]
    labels = set(t0.get("labels") or t0.get("x") or [])
    assert {"Low", "Medium", "High"} <= labels, (
        f"expected Low/Medium/High priority labels, got {labels} — likely charted by date again"
    )


@test("sg_office_charts")
def test_boolean_filter_split_produces_two_slices_not_raw_field(h):
    """Live bug: a query split into two boolean-filtered calls
    (workflow_started=True/False) had no way to tell which call a row
    came from — provenance backfill only handled string filter values,
    not booleans — so the chart fell back to grouping by an unrelated
    field (seen live: a 72-slice pie of sender names). Fixed by adding
    _BOOLEAN_PROVENANCE_LABELS in tools.py's tool_executor_node.

    Label is "In Workflow", not "Started" — matches db_queries.py's own
    pre-existing CASE expression for this column (list_sg_office_emails'
    workflow_label); the router sometimes fetches this same data via that
    native field instead of the boolean filter, and an earlier version of
    this fix used a guessed label ("Started") that disagreed with it."""
    r = h.run_turn("Create a pie chart showing emails with workflow started versus Not Started.", [])
    assert r.chart_data, "expected a chart"
    t0 = r.chart_data["plotly_data"][0]
    labels = set(t0.get("labels") or t0.get("x") or [])
    assert labels == {"In Workflow", "Not Started"}, (
        f"expected exactly 2 slices (In Workflow/Not Started), got {len(labels)}: {labels}"
    )


@test("sg_office_charts")
def test_boolean_filter_trend_splits_into_two_series(h):
    """Same root cause as the pie-chart case above, but for a date-bucket
    trend: "unread and read email trends" needs two separate lines, not
    one merged count. Fixed in chart_engine._extract_date_bucket_chart by
    splitting on a 2-valued boolean-provenance label when present."""
    r = h.run_turn("Create a line chart showing unread and read email trends for the current month.", [])
    assert r.chart_data, "expected a chart"
    names = {t.get("name") for t in r.chart_data["plotly_data"]}
    assert names == {"Read", "Unread"}, f"expected exactly 'Read' and 'Unread' series, got: {names}"


# ---------------------------------------------------------------------------
# Category: long_conversation — a single 15-turn conversation mixing five
# different domains (projects, tasks, meetings, memos, emails, budgets),
# asserting ground truth at checkpoints throughout. Everything else in
# this suite tests 1-3 turn conversations; this is the one test that
# stress-tests context dilution over a realistically long chat — the
# original client complaint from the very start of this engagement was
# "testing everything in one chat causes hallucination", and that
# scenario was never directly verified until now. Checkpoints near the
# end deliberately re-test earlier fixes (subject-count, context-reset,
# boolean-provenance split) under long-history conditions, not just the
# short conversations they were originally verified against.
# ---------------------------------------------------------------------------

@test("long_conversation")
def test_15_turn_mixed_domain_conversation_stays_accurate(h):
    with app.pg_conn() as conn:
        total_projects = ground_truth_total_projects(conn)
        alzaabi_count = ground_truth_project_count(conn, "abdalla alzaabi")
        rashid_count = ground_truth_project_count(conn, "rashid")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sg_office_meetingrequest")
            total_meetings = cur.fetchone()[0]

    history = []
    failures = []

    def check(label, condition, detail):
        if not condition:
            failures.append(f"{label}: {detail}")

    # Turn 1-2: baseline project facts
    r = h.run_turn("list all projects", history); history = r.history
    check("t1_list_all_projects", len(r.tool_results) == total_projects,
          f"expected {total_projects} rows, got {len(r.tool_results)}")

    r = h.run_turn("how many projects does Abdalla Khalid manage", history); history = r.history

    # Turn 3-4: a chart, then a style-only follow-up
    r = h.run_turn("create a bar chart of projects vs status", history); history = r.history
    check("t3_chart_produced", r.chart_data is not None, "expected a chart")

    r = h.run_turn("change the colors", history); history = r.history
    check("t4_chart_still_produced", r.chart_data is not None, "expected the chart to persist through a style-only follow-up")

    # Turn 5: switch domain entirely — tasks
    r = h.run_turn("list all tasks", history); history = r.history
    check("t5_no_chart_for_plain_list", r.chart_data is None, "a plain list should not produce a chart")

    # Turn 6: switch domain again — SG office meetings readiness
    r = h.run_turn("generate a chart of meetings vs readiness status", history); history = r.history
    if r.chart_data:
        names = {t.get("name") for t in r.chart_data["plotly_data"]}
        check("t6_readiness_two_series", "Facility Readiness" in names and "Visitor Readiness" in names,
              f"expected both readiness series, got {names}")
    else:
        failures.append("t6_readiness_two_series: expected a chart, got none")

    # Turn 7: switch domain again — memos
    r = h.run_turn("How many memos are in each status? Provide every status separately.", history); history = r.history

    # Turn 8: re-verify today's boolean-provenance fix survives under long history
    r = h.run_turn("Create a pie chart showing emails with workflow started versus Not Started.", history); history = r.history
    if r.chart_data:
        t0 = r.chart_data["plotly_data"][0]
        labels = set(t0.get("labels") or t0.get("x") or [])
        check("t8_workflow_pie_two_slices", labels == {"In Workflow", "Not Started"},
              f"expected exactly In Workflow/Not Started, got {labels} — boolean-provenance split may not survive long history")
    else:
        failures.append("t8_workflow_pie_two_slices: expected a chart, got none")

    # Turn 9: budgets
    r = h.run_turn("show me a chart of project budgets", history); history = r.history
    if r.chart_data:
        names = {t.get("name") for t in r.chart_data["plotly_data"]}
        check("t9_budget_three_series", {"Allocated", "Spent", "Remaining"} <= names, f"got {names}")
    else:
        failures.append("t9_budget_three_series: expected a chart, got none")

    # Turn 10-11: filtered meetings, then a bare continuation reference
    r = h.run_turn("list all upcoming external meetings", history); history = r.history
    upcoming_count = len(r.tool_results)

    r = h.run_turn("Please show them only", history); history = r.history
    check("t11_continuation_keeps_filter", len(r.tool_results) == upcoming_count,
          f"'show them only' should keep the {upcoming_count}-row upcoming filter, got {len(r.tool_results)} rows")

    # Turn 12: re-verify today's context-reset fix survives under long history —
    # this is the exact "all external meetings" bug fixed earlier, now
    # tested 11 turns deep instead of in a fresh 2-turn conversation.
    r = h.run_turn("Create a chart of all external meetings", history); history = r.history
    check("t12_all_meetings_resets", len(r.tool_results) == total_meetings,
          f"'all external meetings' after a filtered turn should fetch all {total_meetings}, got {len(r.tool_results)} "
          f"— context-reset fix may not hold under long history")

    # Turn 13-14: jump back to projects after 11 turns of unrelated domains
    r = h.run_turn(
        "based on the projects i want to see the number of projects abdalla Alzaabi and rashid are included in. "
        "I want the x axis to represent their name, while the y axis represent the number of projects they are included in",
        history,
    )
    history = r.history
    if r.chart_data:
        t0 = r.chart_data["plotly_data"][0]
        labels = t0.get("x") or t0.get("labels") or []
        values = t0.get("y") or t0.get("values") or []
        by_label = {str(l).lower(): v for l, v in zip(labels, values)}
        alzaabi_val = next((v for k, v in by_label.items() if "alzaabi" in k), None)
        rashid_val = next((v for k, v in by_label.items() if "rashid" in k), None)
        check("t13_subject_count_still_correct",
              alzaabi_val == alzaabi_count and rashid_val == rashid_count,
              f"expected Alzaabi={alzaabi_count}, Rashid={rashid_count}, got Alzaabi={alzaabi_val}, Rashid={rashid_val} "
              f"— 11 turns of unrelated domains in between may have corrupted this")
    else:
        failures.append("t13_subject_count_still_correct: expected a chart, got none")

    r = h.run_turn("list all projects", history); history = r.history
    check("t14_list_all_projects_still_correct", len(r.tool_results) == total_projects,
          f"after 13 turns across 5 domains, 'list all projects' should still fetch all {total_projects} fresh, "
          f"got {len(r.tool_results)} — this is the core 'testing everything in one chat causes hallucination' scenario")

    # Turn 15: one more domain switch, re-verifying the weekly-actions fix
    r = h.run_turn("Create a bar chart showing overdue Weekly Actions by owner.", history); history = r.history
    if r.chart_data:
        t0 = r.chart_data["plotly_data"][0]
        labels = t0.get("x") or t0.get("labels") or []
        check("t15_weekly_action_owner_labels", all(not str(l).isdigit() for l in labels),
              f"expected real owner names, got what looks like raw indices: {labels}")

    assert not failures, "Long-conversation checkpoint failures:\n" + "\n".join(f"  - {f}" for f in failures)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0, help="seconds to sleep between test cases (rate-limit safety)")
    args = parser.parse_args()
    VERBOSE = args.verbose

    categories = sorted({c for c, _, _ in REGISTRY})
    if args.list_categories:
        print("Categories:", ", ".join(categories))
        return

    to_run = [(c, n, f) for c, n, f in REGISTRY if not args.category or c == args.category]
    if not to_run:
        print(f"No tests matched category {args.category!r}. Available: {categories}")
        sys.exit(1)

    h = Harness()
    results = {}  # category -> [(name, passed, msg)]
    for i, (category, name, fn) in enumerate(to_run):
        print(f"[{i+1}/{len(to_run)}] {category}.{name} ...", end=" ", flush=True)
        try:
            fn(h)
            print("PASS")
            results.setdefault(category, []).append((name, True, ""))
        except AssertionError as e:
            print("FAIL")
            print(f"    {e}")
            results.setdefault(category, []).append((name, False, str(e)))
        except Exception as e:
            print("ERROR")
            print(f"    {type(e).__name__}: {e}")
            results.setdefault(category, []).append((name, False, f"{type(e).__name__}: {e}"))
        if i < len(to_run) - 1:
            time.sleep(args.pace)

    print()
    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    total_pass, total_count = 0, 0
    any_fail = False
    for category in sorted(results):
        cases = results[category]
        passed = sum(1 for _, ok, _ in cases if ok)
        total_pass += passed
        total_count += len(cases)
        marker = "OK" if passed == len(cases) else "FAIL"
        if passed != len(cases):
            any_fail = True
        print(f"  [{marker}] {category}: {passed}/{len(cases)}")
        for name, ok, msg in cases:
            if not ok:
                print(f"         - {name}: {msg}")
    print("-" * 70)
    print(f"TOTAL: {total_pass}/{total_count}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
