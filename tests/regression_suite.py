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
from db_queries import list_projects
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
