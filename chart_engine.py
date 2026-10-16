"""
Chart detection and visualization JSON generation for EHCD Chatbot.
Inspects tool results and user query to determine if a chart/graph is appropriate,
then generates Chart.js-compatible JSON configuration.
"""

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Brown color palette as specified
BROWN_COLORS = [
    "#F1DD83",  # Sunlight
    "#808036",  # Linen
    "#A7997C",  # Taupe
    "#7D8C7F",  # Mangrove
    "#917050",  # Earth
    "#E99B67",  # Muted Green
    "#000000",  # Black
]

#BROWN_COLORS = [
#    "#8B4513",  # SaddleBrown
#    "#A0522D",  # Sienna
#    "#CD853F",  # Peru
#    "#DEB887",  # BurlyWood
#    "#D2691E",  # Chocolate
#    "#BC8F8F",  # RosyBrown
#    "#F4A460",  # SandyBrown
#    "#DAA520",  # GoldenRod
#    "#B8860B",  # DarkGoldenRod
#    "#D2B48C",  # Tan
#]

# Light-to-dark gradients anchored on the federal guideline secondary
# colors (Secondary 03 Crimson #C8102E, Secondary 02 Green #00843D) — the
# exact brand hex is kept as one of the 7 steps, not just approximated.
CRIMSON_COLORS = [
    "#F4CFD5", "#E693A1", "#D8586D", "#C8102E", "#A00D25", "#780A1C", "#500612",
]
GREEN_COLORS = [
    "#CCE6D8", "#8CC8A8", "#4CA977", "#00843D", "#006A31", "#004F25", "#003518",
]

PALETTES = {
    "brown": BROWN_COLORS,
    "crimson": CRIMSON_COLORS,
    "green": GREEN_COLORS,
}
DEFAULT_PALETTE = "brown"

# Anchor hex for any OTHER color name a user might ask for in chat (e.g.
# "a bar chart of projects in blue") — gradients for these are generated
# on the fly with the same tint/shade formula used for CRIMSON_COLORS/
# GREEN_COLORS above, not hardcoded as full 7-step lists like those two.
_NAMED_COLOR_HEX = {
    "blue": "#1565C0", "navy": "#0D47A1", "purple": "#6A1B9A", "violet": "#6A1B9A",
    "indigo": "#283593", "orange": "#E65100", "yellow": "#F9A825", "gold": "#B68A35",
    "pink": "#C2185B", "magenta": "#AD1457", "teal": "#00695C", "turquoise": "#00897B",
    "cyan": "#00838F", "gray": "#424242", "grey": "#424242", "black": "#212121",
    "white": "#9E9E9E", "lime": "#827717",
}


def _tint(rgb: tuple, t: float) -> tuple:
    return tuple(round(c + (255 - c) * t) for c in rgb)


def _shade(rgb: tuple, t: float) -> tuple:
    return tuple(round(c * (1 - t)) for c in rgb)


def _build_gradient(hex_color: str) -> List[str]:
    """Light-to-dark 7-step gradient anchored on one hex color — the same
    formula CRIMSON_COLORS/GREEN_COLORS were generated with, run live here
    for any other color name instead of hardcoding every possible one."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    steps = [
        _tint(rgb, 0.80), _tint(rgb, 0.55), _tint(rgb, 0.30),
        rgb,
        _shade(rgb, 0.20), _shade(rgb, 0.40), _shade(rgb, 0.60),
    ]
    return ["#%02X%02X%02X" % s for s in steps]


def resolve_palette_colors(palette: str) -> List[str]:
    """Named-palette lookup first (brown/crimson/green — the fixed, fully
    pre-defined ones), then fall back to generating a gradient for any
    other recognized color name, then brown if the name isn't recognized
    at all."""
    if palette in PALETTES:
        return PALETTES[palette]
    anchor = _NAMED_COLOR_HEX.get(palette)
    if anchor:
        return _build_gradient(anchor)
    return BROWN_COLORS


_CHART_WORD_RE = re.compile(
    r"\b(bar\s?chart|pie\s?chart|line\s?chart|chart|graphs?|visualiz\w*|diagram\w*|pie)\b",
    re.IGNORECASE,
)

# Appended (invisible) to a saved assistant turn's history text whenever
# that turn included a chart — read back on the next turn so a color/type
# -only follow-up ("show in orange colour") can still be recognized as a
# chart request without repeating the word "chart" itself. Pure zero-width
# characters, no readable text — a readable bracket tag like "[chart]" was
# tried first and the model sometimes echoed it back into its own visible
# answer on the next turn, since it's replayed to it as prior conversation.
CHART_SHOWN_MARKER = "​‌‍​"


def recent_turn_had_chart(conversation_history: list) -> bool:
    """True if the most recent assistant turn in history included a chart."""
    for turn in reversed(conversation_history or []):
        if turn.get("role") == "assistant":
            return CHART_SHOWN_MARKER in (turn.get("content") or "")
    return False


def last_user_query(conversation_history: list) -> str:
    """The most recent USER turn's text, for detect_chart_opportunity's
    previous_query — called before this turn's own query is appended."""
    for turn in reversed(conversation_history or []):
        if turn.get("role") == "user":
            return turn.get("content") or ""
    return ""


# Follow-up phrasing that only makes sense as "redo the chart I just saw,
# differently" — a bare chart word isn't in these ("show in orange colour"
# never says "chart"), so they only count when the PRIOR turn had a chart
# (see CHART_SHOWN_MARKER / recent_had_chart below). Without that guard
# these words are too common in unrelated sentences to trust alone.
_CHART_CONTINUATION_RE = re.compile(
    r"\b(colou?r|instead|switch (it|this) to|change (it|this) to|as an?\s*(line|bar|pie))\b",
    re.IGNORECASE,
)


def is_chart_request(query: str, recent_had_chart: bool = False, want_chart: Optional[bool] = None) -> bool:
    """
    True if a chart/graph is wanted, independent of whether chartable data
    actually exists. Shared with app.py so it can tell when the model's
    response claims a chart is coming but detect_chart_opportunity ended
    up unable to build one (e.g. the requested field doesn't exist for
    that entity, like a task's "budget") — the model has no way to know
    that in advance, since chart detection runs after its response is
    already generated.

    want_chart — the ROUTER's own decision (see _CHART_INTENT_PARAMS in
    tools.py), passed through from the tool call it already made. This is
    authoritative and wins outright when given: the router read the whole
    conversation and decided directly, so there's nothing left to guess.
    Falls back to the keyword/regex heuristics below only when the caller
    doesn't have a router decision to pass (want_chart is None) — kept as
    a safety net, not the primary path anymore, because it kept missing
    real phrasings no fixed keyword list can fully anticipate: "change
    the colours" (plural), "x axis should be name" (never says "chart"
    at all), a style tweak with no chart word of its own.

    Uses word-boundary matching, not plain substring — a naive `"pie" in
    query` check previously fired on any word containing "pie" as a
    substring, e.g. "recipients" ("reci-PIE-nts"), turning an ordinary
    retrieval question into a false chart request.
    """
    if want_chart is not None:
        return want_chart

    query_lower = query.lower()
    if _CHART_WORD_RE.search(query_lower):
        return True
    if "compare budget" in query_lower:
        return True
    return recent_had_chart and bool(_CHART_CONTINUATION_RE.search(query_lower))


# A few extra words alias to one of the two brand secondary palettes
# (people say "red" more often than "crimson"); every other word here maps
# to itself and gets its gradient built on the fly by resolve_palette_colors.
_COLOR_WORD_ALIASES = {"red": "crimson", "maroon": "crimson", "violet": "purple", "grey": "gray"}
_COLOR_WORDS = set(PALETTES) | set(_NAMED_COLOR_HEX) | set(_COLOR_WORD_ALIASES)
_COLOR_WORD_RE = re.compile(r"\b(" + "|".join(sorted(_COLOR_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


_COLORLESS_CHANGE_RE = re.compile(r"\bcolou?rs?\b", re.IGNORECASE)

# Offered in rotation for a colorless "change the colors" follow-up — must
# stay different from whatever was used last, or the request is pointless.
_PALETTE_ROTATION = ["crimson", "green", "blue", "purple", "teal", "orange"]
_PALETTE_MARKER_BASE = 0xFE00  # invisible Unicode variation selectors


def _encode_palette_marker(palette_name: str) -> str:
    """Invisible single character recording which palette this turn's
    chart used, appended to saved history text (same technique as
    CHART_SHOWN_MARKER) so a later colorless "change the colors" knows
    what to rotate away FROM. Empty string for a palette not in the
    rotation (e.g. an explicitly-named one-off like "blue") — nothing to
    rotate away from in that case."""
    if palette_name not in _PALETTE_ROTATION:
        return ""
    return chr(_PALETTE_MARKER_BASE + _PALETTE_ROTATION.index(palette_name))


def _last_used_palette(conversation_history: list) -> Optional[str]:
    """The palette encoded on the most recent assistant turn that had a
    chart, or None if there wasn't one / it used a non-rotation palette."""
    for turn in reversed(conversation_history or []):
        if turn.get("role") == "assistant":
            for ch in turn.get("content") or "":
                cp = ord(ch)
                if _PALETTE_MARKER_BASE <= cp < _PALETTE_MARKER_BASE + len(_PALETTE_ROTATION):
                    return _PALETTE_ROTATION[cp - _PALETTE_MARKER_BASE]
            return None
    return None


def detect_palette_from_query(query: str, conversation_history: Optional[list] = None) -> str:
    """Pick a palette from ANY color word in the query itself (e.g. "a bar
    chart of projects in blue") — not just the 3 named brand palettes.
    Word-boundary matching, not substring, for the same reason as
    is_chart_request above (a naive "red" in query check would fire on
    "prepared", "credit", etc).

    A colorless "change the colors" follow-up (no color word of its own)
    rotates to a genuinely different palette than whichever one the most
    recent chart actually used, read from conversation_history — without
    this it always fell back to the same default (brown) on every repeat,
    so asking "change the colors" three times in a row produced the exact
    same chart three times. Pass conversation_history to enable this;
    omit it to keep the plain no-color-word default (the frontend's own
    palette toggle doesn't need this at all — it re-colors client-side
    from chart_data's `available_palettes`, no backend call involved).
    """
    m = _COLOR_WORD_RE.search(query)
    if m:
        word = m.group(1).lower()
        return _COLOR_WORD_ALIASES.get(word, word)

    if conversation_history is not None and _COLORLESS_CHANGE_RE.search(query):
        last = _last_used_palette(conversation_history)
        idx = (_PALETTE_ROTATION.index(last) + 1) % len(_PALETTE_ROTATION) if last in _PALETTE_ROTATION else 0
        return _PALETTE_ROTATION[idx]

    return DEFAULT_PALETTE


def _palette_fields(colors: List[str], used_count: int, palette_name: str = DEFAULT_PALETTE) -> Dict[str, Any]:
    """The colors actually used for this chart's traces (a flat list, in
    trace order) plus the full catalog of selectable palettes — so a
    frontend palette-switcher doesn't need any colors hardcoded itself.
    If the chart was drawn in a color outside the 3 fixed brand palettes
    (e.g. the user asked for "blue" in chat), that gradient is appended
    too, so a follow-up toggle for THIS chart can switch back to it."""
    available = [{"name": name, "colors": c} for name, c in PALETTES.items()]
    if palette_name not in PALETTES:
        available.append({"name": palette_name, "colors": colors})
    return {
        "palette": colors[:max(used_count, 1)],
        "available_palettes": available,
    }


def detect_chart_opportunity(
    query: str,
    tool_results: List[Dict],
    assistant_text: str,
    palette: str = DEFAULT_PALETTE,
    recent_had_chart: bool = False,
    previous_query: str = "",
    want_chart: Optional[bool] = None,
    router_chart_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze query + tool results to determine if a chart is appropriate.
    Returns chart_data dict or None.

    `palette` picks which named color set (see PALETTES) draws the chart —
    defaults to the brand's primary brown palette; pass "crimson" or
    "green" for either secondary palette instead.

    `want_chart`/`router_chart_type` — the ROUTER's own decision (see
    _CHART_INTENT_PARAMS in tools.py), from the tool call it already
    made. Authoritative when given: it read the whole conversation and
    decided directly, so nothing here needs to re-guess "should we chart
    at all" or "which type" from keyword patterns. `recent_had_chart`/
    `previous_query`-based heuristics below are the fallback for when no
    router decision is available, not the primary path.

    `recent_had_chart` — see is_chart_request — lets a color/type-only
    follow-up to a chart still count as a chart request (fallback path).

    `previous_query` — the prior turn's query text, used only when THIS
    turn names no field of its own (see _extract_chart_data) and, for
    chart TYPE only, blended in alongside this turn's words so "show in
    orange" after a pie chart still renders as a pie, not the bar default.
    """
    query_lower = query.lower()
    colors = resolve_palette_colors(palette)
    # For TYPE inference only — safe to blend broadly (see
    # _query_names_chart_type below: this turn's own explicit type
    # always wins regardless of what's blended in).
    type_fallback_query_lower = previous_query.lower() if recent_had_chart and previous_query else None
    # For FIELD matching — much narrower. Reusing the previous turn's
    # wording must only happen for a genuinely content-free style
    # continuation ("show in golden colour") — not for any query that
    # merely fails to name a recognized FIELD, since a fully-formed new
    # request ("email trends for the current month") can legitimately do
    # that too (trend/month aren't field names) and must never be
    # silently replaced by stale, unrelated wording from an earlier turn.
    # The router's own want_chart (when given) replaces the regex as the
    # "is this stylistically a continuation" signal — it already knows,
    # from the full conversation, whether this turn is a bare style tweak
    # rather than a new self-contained request.
    is_style_continuation_signal = (
        want_chart if want_chart is not None else bool(_CHART_CONTINUATION_RE.search(query_lower))
    )
    is_pure_style_continuation = (
        recent_had_chart
        and is_style_continuation_signal
        and not _CHART_WORD_RE.search(query_lower)
        and "compare budget" not in query_lower
        and not _DATE_BUCKET_RE.search(query_lower)
        and "readiness" not in query_lower
    )
    field_fallback_query_lower = previous_query.lower() if is_pure_style_continuation and previous_query else None

    if not is_chart_request(query, recent_had_chart, want_chart):
        return None

    # Most specific case first: a query naming BOTH a single budget metric
    # (e.g. "spent") AND a grouping dimension (e.g. "by department") needs
    # one trace per group, each covering only its own projects — a shape
    # the shared labels/datasets pipeline below (one flat series per metric,
    # all projects on one shared x-axis) can't produce. Handled separately
    # and returns its own complete payload; falls through to the normal
    # cases if the query isn't this specific or the grouped data is empty.
    grouped_chart = _extract_grouped_budget_chart(query_lower, tool_results, colors, palette, router_chart_type)
    if grouped_chart:
        return grouped_chart

    # Try to extract chartable data from tool results
    chart_data = _extract_chart_data(query_lower, tool_results, field_fallback_query_lower)
    if not chart_data:
        return None

    labels = chart_data["labels"]
    datasets = chart_data["datasets"]
    # The router's own type (when given) wins outright — it's the same
    # authoritative decision as want_chart, not a guess. Otherwise this
    # turn's own explicit type wins over blended-in previous-turn text
    # (see _query_names_chart_type), so a stale "line" from before can
    # never override an explicit "bar" asked for now.
    if router_chart_type:
        chart_type = router_chart_type
    elif chart_data.get("chart_type"):
        chart_type = chart_data["chart_type"]
    elif _query_names_chart_type(query_lower):
        chart_type = _infer_chart_type(query_lower, labels)
    else:
        type_query_lower = f"{type_fallback_query_lower} {query_lower}" if type_fallback_query_lower else query_lower
        chart_type = _infer_chart_type(type_query_lower, labels)

    if len(labels) < 1:
        return None

    # Generate Plotly config (data + layout, no hardcoded container ID)
    plotly_config = _build_plotly_config(chart_type, labels, datasets, colors)

    result = {
        "chart_type": chart_type,
        "plotly_data": plotly_config["data"],
        "plotly_layout": plotly_config["layout"],
        "audit": {
            "total_items": len(labels),
            "labels": labels[:20],
        },
        **_palette_fields(colors, max(len(labels), len(datasets)), palette),
    }
    if chart_data.get("no_budget_entities_present"):
        result["no_budget_entities_present"] = True
    return result


_METRIC_KEYWORDS = [
    (["spent", "expenditure", "expense"], "spent_budget", "Spent Budget"),
    (["remaining", "unused", "budget left"], "budget_left", "Remaining Budget"),
    (["allocated", "allocation"], "allocated_budget", "Allocated Budget"),
]


def _detect_budget_metrics(query_lower: str) -> List[tuple]:
    """Returns [(field_name, display_label), ...] for every budget metric
    the query names — e.g. "spent vs allocated" returns both. Empty if the
    query doesn't name any specific one (e.g. just "budget" generally),
    which leaves the existing all-three-metrics comparison to handle it."""
    return [
        (field, label)
        for keywords, field, label in _METRIC_KEYWORDS
        if any(k in query_lower for k in keywords)
    ]


# Fields that identify WHICH row a chart item is (already used as the
# x-axis label elsewhere) — never valid as a group-by/metric axis
# themselves, so excluded from the generic field matching below.
_ENTITY_NAME_FIELDS = frozenset({
    "project_name_en", "project_name_ar", "sg_office_name_en", "sg_office_name_ar",
    "task_name", "resolution_topic_en", "resolution_topic_ar", "subject", "code",
})

# Field-norm -> extra words a user might say instead of the real column
# name — e.g. category_name means "department"/"sector" to most users;
# emails' own status-equivalent column is literally named "workflow".
_FIELD_SYNONYMS = {
    "category": ["department", "sector"],
    "workflow": ["status"],
    "readiness status": ["confirmation status", "visitor status"],
}

_FIELD_SUFFIXES = ("_label", "_en", "_ar", "_name")
_FIELD_SUFFIX_RANK = {"_label": 0, "_en": 1, "_name": 2, "_ar": 3}


def _normalize_field(key: str) -> str:
    """'status_label' -> 'status', 'category_name' -> 'category' — so a DB
    column reads like the plain word a user would actually type."""
    for suf in _FIELD_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)].replace("_", " ").strip()
    return key.replace("_", " ").strip()


def _field_is_numeric(key: str, tool_results: List[Dict]) -> bool:
    values = [r.get(key) for r in tool_results if isinstance(r, dict) and r.get(key) is not None]
    return bool(values) and all(_to_float(v) is not None for v in values)


def _raw_code_keys(keys, tool_results: List[Dict]) -> set:
    """Fields that are raw numeric codes standing in for some OTHER field
    with the same normalized name that actually holds real text — e.g.
    status/status_en/status_ar all normalize to "status", but only
    status_en is ever translated; status_ar keeps the same raw numeric
    code and is never resolved. Trusting the _ar suffix alone (as if it
    always meant "resolved Arabic text") let that raw code slip through
    as if it were a real metric. Grouping by normalized name and checking
    actual values instead catches this regardless of which field (if
    any) ends up holding the real text."""
    groups: Dict[str, List[str]] = {}
    for k in keys:
        groups.setdefault(_normalize_field(k), []).append(k)
    drop = set()
    for group_keys in groups.values():
        if len(group_keys) < 2:
            continue
        if any(not _field_is_numeric(k, tool_results) for k in group_keys):
            drop.update(k for k in group_keys if _field_is_numeric(k, tool_results))
    return drop


def _match_field(
    query_lower: str, tool_results: List[Dict], numeric: Optional[bool] = None,
    exclude: frozenset = frozenset(),
) -> Optional[str]:
    """
    Generic axis detection: find a real field in tool_results whose plain
    name (or a known synonym) is actually named in the query — e.g.
    "priority" matches a real priority_label column — so any DB field
    works as a chart axis without a hand-added keyword each time.
    numeric=True/False restricts to a metric (y-axis) vs. a categorical
    (x-axis/group-by) field; None allows either.
    """
    keys = set()
    for row in tool_results:
        if isinstance(row, dict):
            keys.update(row.keys())
    keys -= _ENTITY_NAME_FIELDS
    keys -= exclude
    keys = {k for k in keys if k != "id" and not k.endswith("_id")}
    # A raw code (e.g. request_type=2) superseded by a sibling field that
    # actually holds real text is never a real axis itself — same rule
    # tools.py uses to drop these from the table, so it's never picked as
    # a "numeric metric" over the category it encodes.
    keys -= _raw_code_keys(keys, tool_results)

    best_key, best_score = None, None
    for key in keys:
        norm = _normalize_field(key)
        if not norm:
            continue
        # Plain "+s" plural too (sender -> senders) — "top 10 email
        # senders" otherwise never matched sender_name at all, since the
        # word-boundary match requires the exact word, not a substring.
        words = [norm, f"{norm}s"] + _FIELD_SYNONYMS.get(norm, [])
        if not any(re.search(r"\b" + re.escape(w) + r"\b", query_lower) for w in words):
            continue
        if numeric is not None and _field_is_numeric(key, tool_results) != numeric:
            continue
        if numeric is False:
            # A true identity field (e.g. a meeting's own name) is never a
            # real grouping dimension — every row has its own distinct
            # value, so nothing repeats. Skip it instead of letting a
            # longer field name like "meeting" out-score "status" purely
            # on string length.
            vals = [r.get(key) for r in tool_results if isinstance(r, dict) and r.get(key)]
            if vals and len(set(map(str, vals))) >= len(vals):
                continue
        # "status" is almost always the actual intent even next to a
        # longer word — "facility readiness status" means group by status
        # (facility is just which readiness), not facility type, so a
        # bare "status"/"readiness status" match always wins over a
        # longer but unrelated word purely on string length.
        is_status = norm in ("status", "readiness status")
        # Prefer the most specific (longest) name match; among ties prefer
        # the most-resolved column (a _label beats a bare/_ar twin).
        rank = next((r for suf, r in _FIELD_SUFFIX_RANK.items() if key.endswith(suf)), 4)
        score = (is_status, len(norm), -rank)
        if best_score is None or score > best_score:
            best_key, best_score = key, score
    return best_key


def _entity_display_name(item: Dict) -> Optional[str]:
    return (
        item.get("project_name_en") or item.get("sg_office_name_en")
        or item.get("task_name") or item.get("resolution_topic_en")
    )


def _extract_grouped_budget_chart(
    query_lower: str, tool_results: List[Dict], colors: List[str] = BROWN_COLORS,
    palette_name: str = DEFAULT_PALETTE, router_chart_type: Optional[str] = None,
) -> Optional[Dict]:
    """
    Handle queries naming a budget metric (or several, e.g. "spent vs
    allocated") together with a grouping dimension (category/department,
    manager, or status), e.g. "line chart of budget spent on each project
    segmented by department". Returns the complete chart payload directly.

    Single metric -> one trace per group, each plotting only the projects
    in that group (keeps per-project detail — good for spotting which
    project drives a group's total).
    Multiple metrics -> one trace per METRIC instead, x-axis is the groups
    themselves, values summed per group (comparing metrics side by side
    across groups is the more readable shape once there's more than one
    number per group; per-project-by-metric would be metrics × projects
    traces, unreadable past a handful of projects).
    """
    if not tool_results:
        return None

    metrics = _detect_budget_metrics(query_lower)
    group_key = _match_field(query_lower, tool_results, numeric=False)
    if not metrics or not group_key:
        return None

    chart_type = router_chart_type or _infer_chart_type(query_lower, [])
    dashed = any(w in query_lower for w in ["dotted", "dashed"])

    layout = {
        "title": "",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "legend": {"orientation": "h", "y": -0.2},
    }

    if len(metrics) == 1:
        metric_field, metric_label = metrics[0]
        rows = []
        for item in tool_results:
            name = item.get("project_name_en") or item.get("sg_office_name_en")
            val = _to_float(item.get(metric_field))
            if name and val is not None:
                group = str(item.get(group_key) or "Uncategorized")
                rows.append((name, group, val))

        if not rows:
            return None

        grouped: Dict[str, List] = {}
        for name, group, val in rows:
            grouped.setdefault(group, []).append((name, val))

        if chart_type == "pie":
            # A pie has no x-axis / per-project concept — sum per group.
            pie_labels = list(grouped.keys())
            pie_values = [sum(v for _, v in items) for items in grouped.values()]
            traces = [{
                "labels": pie_labels,
                "values": pie_values,
                "type": "pie",
                "marker": {"colors": colors[:len(pie_labels)]},
                "name": metric_label,
            }]
        else:
            traces = []
            for i, (group_name, items) in enumerate(grouped.items()):
                color = colors[i % len(colors)]
                x_values = [n for n, _ in items]
                y_values = [v for _, v in items]
                if chart_type == "line":
                    line_style = {"color": color}
                    if dashed:
                        line_style["dash"] = "dot"
                    traces.append({
                        "x": x_values, "y": y_values,
                        "type": "scatter", "mode": "lines+markers",
                        "name": group_name, "line": line_style,
                    })
                else:
                    traces.append({
                        "x": x_values, "y": y_values,
                        "type": "bar", "name": group_name,
                        "marker": {"color": color},
                    })
            layout["xaxis"] = {"title": "Project"}
            layout["yaxis"] = {"title": metric_label}
            if chart_type == "bar":
                layout["barmode"] = "group"

        all_labels = [name for name, _, _ in rows]
        audit = {
            "total_items": len(rows),
            "labels": all_labels[:20],
            "metric_fields": [metric_field],
            "group_field": group_key,
            "groups": list(grouped.keys()),
        }
    else:
        # Multiple metrics: aggregate to one total per group per metric.
        group_totals: Dict[str, Dict[str, float]] = {}
        for item in tool_results:
            if not (item.get("project_name_en") or item.get("sg_office_name_en")):
                continue
            group = str(item.get(group_key) or "Uncategorized")
            for metric_field, _ in metrics:
                val = _to_float(item.get(metric_field))
                if val is not None:
                    bucket = group_totals.setdefault(group, {})
                    bucket[metric_field] = bucket.get(metric_field, 0) + val

        if not group_totals:
            return None

        group_names = list(group_totals.keys())

        if chart_type == "pie":
            # Pie can only show one metric's proportions — use the first
            # named one rather than trying to merge several into one ring.
            metric_field, metric_label = metrics[0]
            pie_values = [group_totals[g].get(metric_field, 0) for g in group_names]
            traces = [{
                "labels": group_names,
                "values": pie_values,
                "type": "pie",
                "marker": {"colors": colors[:len(group_names)]},
                "name": metric_label,
            }]
        else:
            traces = []
            for i, (metric_field, metric_label) in enumerate(metrics):
                color = colors[i % len(colors)]
                y_values = [group_totals[g].get(metric_field, 0) for g in group_names]
                if chart_type == "line":
                    line_style = {"color": color}
                    if dashed:
                        line_style["dash"] = "dot"
                    traces.append({
                        "x": group_names, "y": y_values,
                        "type": "scatter", "mode": "lines+markers",
                        "name": metric_label, "line": line_style,
                    })
                else:
                    traces.append({
                        "x": group_names, "y": y_values,
                        "type": "bar", "name": metric_label,
                        "marker": {"color": color},
                    })
            layout["xaxis"] = {"title": _normalize_field(group_key).title()}
            layout["yaxis"] = {"title": "Budget"}
            if chart_type == "bar":
                layout["barmode"] = "group"

        audit = {
            "total_items": len(group_names),
            "labels": group_names[:20],
            "metric_fields": [f for f, _ in metrics],
            "group_field": group_key,
            "groups": group_names,
        }

    used_count = len(traces[0]["labels"]) if chart_type == "pie" else len(traces)
    return {
        "chart_type": chart_type,
        "plotly_data": traces,
        "plotly_layout": layout,
        "audit": audit,
        **_palette_fields(colors, used_count, palette_name),
    }


# "weekly" excludes "weekly action(s)" — that's this app's own entity name
# (the Theyab-workspace module), not a request to bucket by week. Seen
# live: "Weekly Actions by priority" false-triggered this case purely
# because "weekly" appears in the entity's name, producing a date-grouped
# chart instead of the requested priority breakdown — the text answer and
# the actual chart ended up describing two completely different things.
_DATE_BUCKET_RE = re.compile(
    r"\b(per day|daily|each day|per week|weekly(?!\s+actions?)|per month|monthly|trend|trends|over time)\b",
    re.IGNORECASE,
)
_WEEK_GRANULARITY_RE = re.compile(r"\b(per week|weekly(?!\s+actions?)|each week)\b", re.IGNORECASE)
# Checked in this order — any of these present in a row is used as the
# date to bucket by, first match wins.
_DATE_FIELD_CANDIDATES = (
    "received_datetime", "scheduled_date", "date_received", "created_at",
    "meeting_date", "deadline", "closed_at", "updated_at", "due_at",
    "scheduled_on", "readiness_updated_at", "arrived_at",
)


def _parse_date(val) -> Optional[datetime]:
    """Tolerant parse of whatever date/datetime string shape this DB
    returns (date-only, ISO datetime, with/without timezone or trailing
    'Z') — returns None rather than raising on anything unparseable."""
    if not val:
        return None
    s = str(val).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _extract_date_bucket_chart(query_lower: str, tool_results: List[Dict]) -> Optional[Dict]:
    """
    Handle "per day"/"weekly"/"trend"/"over time" style queries by
    counting rows per day/week/month from whichever date field is
    actually present — e.g. "emails received each day for the last 7
    days", "memos received per week". Returns a time-ordered chart, or
    None if the query doesn't ask for this or no usable date field exists.
    """
    if not _DATE_BUCKET_RE.search(query_lower) or not tool_results:
        return None

    keys = set()
    for row in tool_results:
        if isinstance(row, dict):
            keys.update(row.keys())
    date_field = next((f for f in _DATE_FIELD_CANDIDATES if f in keys), None)
    if not date_field:
        return None

    # Bare "month"/"week" (as in "for the current month") describes the
    # DATE RANGE, not the bucket size — bucketing a single month's own
    # data by month always yields exactly one bucket, which the <2 check
    # below then rejects outright. Only explicit "per week"/"monthly"
    # phrasing means bucket-by-that-size; everything else defaults to
    # daily buckets, the only granularity that's ever useful within a
    # single named month/week.
    if _WEEK_GRANULARITY_RE.search(query_lower):
        granularity = "week"
    elif any(p in query_lower for p in ("per month", "monthly", "each month")):
        granularity = "month"
    else:
        granularity = "day"

    def _bucket_key(parsed: datetime) -> str:
        if granularity == "day":
            return parsed.strftime("%Y-%m-%d")
        if granularity == "week":
            return (parsed - timedelta(days=parsed.weekday())).strftime("%Y-%m-%d")
        return parsed.strftime("%Y-%m")

    chart_type = "line" if "trend" in query_lower or granularity != "day" else "bar"

    # Labels backfilled by tool_executor_node's boolean-provenance
    # mechanism (_BOOLEAN_PROVENANCE_LABELS in tools.py) — when exactly 2
    # distinct values are present, this is a two-series trend ("unread
    # and read email trends"), not one merged count. Without this split,
    # two separate is_read=True/False calls' rows land merged into a
    # single misleading "Count" line that can't distinguish them at all.
    for split_key in ("is_read_label", "workflow_started_label", "is_confirmed_label"):
        split_values = sorted({
            row[split_key] for row in tool_results
            if isinstance(row, dict) and row.get(split_key)
        })
        if len(split_values) != 2:
            continue
        buckets_by_value: Dict[str, Dict[str, int]] = {v: {} for v in split_values}
        for row in tool_results:
            parsed = _parse_date(row.get(date_field))
            val = row.get(split_key)
            if not parsed or val not in buckets_by_value:
                continue
            key = _bucket_key(parsed)
            buckets_by_value[val][key] = buckets_by_value[val].get(key, 0) + 1
        all_keys = sorted({k for v in buckets_by_value.values() for k in v})
        if len(all_keys) < 2:
            break  # too sparse to chart as a trend — fall through to the merged series
        return {
            "labels": all_keys,
            "datasets": [
                {"label": v, "data": [buckets_by_value[v].get(k, 0) for k in all_keys]}
                for v in split_values
            ],
            "chart_type": chart_type,
        }

    buckets: Dict[str, int] = {}
    for row in tool_results:
        parsed = _parse_date(row.get(date_field))
        if not parsed:
            continue
        buckets[_bucket_key(parsed)] = buckets.get(_bucket_key(parsed), 0) + 1

    if len(buckets) < 2:
        return None
    sorted_keys = sorted(buckets.keys())
    return {
        "labels": sorted_keys,
        "datasets": [{"label": "Count", "data": [buckets[k] for k in sorted_keys]}],
        "chart_type": chart_type,
    }


_READINESS_RANK = {"Not Confirmed": 0, "Pending": 1, "Confirmed": 2}


def _extract_combined_readiness_chart(query_lower: str, tool_results: List[Dict]) -> Optional[Dict]:
    """
    "meetings vs readiness status" with no "visitor"/"facility" qualifier
    means both halves of readiness — facility task status AND visitor
    readiness — so when the router calls both
    list_sg_office_meeting_facilities and list_sg_office_meeting_visitors
    for it, their rows land merged in one list. One meeting can have
    several facility tasks or visitors, so each is reduced to its single
    WORST (most-blocking) status — any one unconfirmed item means that
    meeting isn't ready — then charted as one grouped bar: meetings on
    the x-axis, Facility/Visitor readiness as two series. A qualified
    question ("visitor readiness"/"facility readiness") is left to the
    per-meeting matcher below, since then only one tool's rows exist.
    """
    if "readiness" not in query_lower:
        return None
    if "visitor" in query_lower or "facility" in query_lower:
        return None

    facility_rows = [r for r in tool_results if isinstance(r, dict) and "facility_label" in r]
    visitor_rows = [r for r in tool_results if isinstance(r, dict) and "readiness_status_label" in r]
    if not facility_rows or not visitor_rows:
        return None

    def _worst_per_meeting(rows, key):
        worst: Dict[str, str] = {}
        for row in rows:
            name = row.get("organization") or row.get("requester")
            status = row.get(key)
            if not name or not status:
                continue
            status = str(status)
            if name not in worst or _READINESS_RANK.get(status, 99) < _READINESS_RANK.get(worst[name], 99):
                worst[name] = status
        return worst

    facility_by_meeting = _worst_per_meeting(facility_rows, "status_label")
    visitor_by_meeting = _worst_per_meeting(visitor_rows, "readiness_status_label")
    meetings = list(dict.fromkeys(list(facility_by_meeting) + list(visitor_by_meeting)))
    if not meetings:
        return None

    return {
        "labels": meetings,
        "datasets": [
            {"label": "Facility Readiness", "data": [facility_by_meeting.get(m) for m in meetings]},
            {"label": "Visitor Readiness", "data": [visitor_by_meeting.get(m) for m in meetings]},
        ],
        # No explicit chart_type — let the caller infer bar/line/pie from
        # the query itself (a hardcoded "bar" here ignored "line chart").
    }


def _extract_readiness_per_meeting_chart(query_lower: str, tool_results: List[Dict]) -> Optional[Dict]:
    """
    "visitor readiness status"/"facility readiness status" means one bar
    PER MEETING with its actual status text as the value (not a count) —
    meetings on the x-axis, the resolved label (Confirmed/Not Confirmed/
    Pending), not the raw numeric code, on the y-axis. Facility/visitor
    rows have no "meeting" field of their own — organization (falling
    back to requester) is what identifies which meeting each belongs to.
    """
    if "readiness" not in query_lower:
        return None
    if "visitor" in query_lower:
        key, series_label = "readiness_status_label", "Visitor Readiness"
    elif "facility" in query_lower:
        key, series_label = "status_label", "Facility Readiness"
    else:
        return None  # bare "readiness" — the combined chart above handles it

    items = []
    for row in tool_results:
        if not isinstance(row, dict):
            continue
        name = row.get("organization") or row.get("requester")
        status = row.get(key)
        if name and status:
            items.append((str(name), str(status)))
    if len(items) < 2:
        return None

    return {
        "labels": [n for n, _ in items],
        "datasets": [{"label": series_label, "data": [s for _, s in items]}],
    }


def _match_field_by_value(query_lower: str, tool_results: List[Dict]) -> Optional[str]:
    """
    Fallback for when no field NAME matched the query: look for a field
    whose actual VALUES are named instead — e.g. "completed, pending and
    unassigned facility tasks" never says "status", but those three words
    ARE real status_label values. Requires 2+ of a field's own distinct
    values to be named, not just one, so an ordinary word that happens to
    equal some value somewhere doesn't false-positive.
    """
    keys = set()
    for row in tool_results:
        if isinstance(row, dict):
            keys.update(k for k in row.keys() if k.endswith(("_label", "_en", "_ar", "_name")))
    keys -= _ENTITY_NAME_FIELDS

    best_key, best_hits = None, 1
    for key in keys:
        values = {str(r.get(key)) for r in tool_results if isinstance(r, dict) and r.get(key)}
        hits = sum(
            1 for v in values
            if re.search(r"\b" + re.escape(v.lower()) + r"\b", query_lower)
        )
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


_COUNT_FRAMING_RE = re.compile(r"\b(number of|how many|count of)\b", re.IGNORECASE)
_PROVENANCE_FILTER_FIELDS = ("status", "request_type", "priority", "readiness_status", "item_type", "project_manager")


def _extract_subject_count_chart(query_lower: str, tool_results: List[Dict]) -> Optional[Dict]:
    """
    "number of projects X is included in" / "...X and Y are included in,
    x axis = name, y axis = number of projects" — a plain count per
    filtered SUBJECT (one or many), not a breakdown across a data field,
    so the usual "2+ distinct values of some row field" distribution
    logic doesn't apply here (there's no such field — the grouping is
    "which filter value actually fetched this row"). tools.py backfills
    a resolved _label onto every row for whichever filter was used on
    that call; grouping by that label and counting rows per group gives
    one bar per subject queried, whether there's one or several.
    """
    if not _COUNT_FRAMING_RE.search(query_lower) or not tool_results:
        return None
    for filt in _PROVENANCE_FILTER_FIELDS:
        key = f"{filt}_label"
        counts: Dict[str, int] = {}
        for r in tool_results:
            if isinstance(r, dict) and r.get(key):
                counts[str(r[key])] = counts.get(str(r[key]), 0) + 1
        if counts:
            return {
                "labels": list(counts.keys()),
                "datasets": [{"label": "Count", "data": list(counts.values())}],
            }
    return None


def _extract_chart_data(
    query_lower: str, tool_results: List[Dict], fallback_query_lower: Optional[str] = None
) -> Optional[Dict]:
    """Extract label-value pairs from tool results for charting.

    fallback_query_lower — the PREVIOUS turn's query, only used when this
    turn names no real field at all (e.g. "show in golden colour" has
    none) — e.g. "bar chart of projects vs status" then "show in golden
    colour" would otherwise default straight to the budget chart (present
    whether or not asked for) instead of re-showing status, since the new
    message alone gives nothing to match. An explicit new field this turn
    ("show it by venue instead") always wins outright — this only fills
    in when there's truly nothing to go on.
    """
    if not tool_results:
        return None

    if fallback_query_lower and not _match_field(query_lower, tool_results):
        query_lower = fallback_query_lower

    # Case 0: "per day"/"weekly"/"trend" style queries — bucketed by date,
    # checked before everything else since these are unambiguous about
    # wanting a time series, not a category breakdown.
    date_chart = _extract_date_bucket_chart(query_lower, tool_results)
    if date_chart:
        return date_chart

    # Case 0b: unqualified "readiness status" — both facility and visitor
    # readiness combined, if both came back this turn.
    readiness_chart = _extract_combined_readiness_chart(query_lower, tool_results)
    if readiness_chart:
        return readiness_chart

    # Case 0c: "visitor"/"facility" readiness — one bar per meeting with
    # its real status text, not an aggregate count.
    per_meeting_chart = _extract_readiness_per_meeting_chart(query_lower, tool_results)
    if per_meeting_chart:
        return per_meeting_chart

    # Case 1: A field explicitly named in the query, resolved generically
    # against whatever columns these rows actually have — e.g. "projects
    # vs status", "tasks by priority", "meetings by venue" — checked
    # FIRST, before the budget case below, since budget fields are present
    # on project/office rows whether or not budget was asked for (e.g.
    # "projects vs status" used to chart budget anyway because every row
    # happens to carry allocated_budget). A new DB column works here with
    # no code change. Gated to rows that carry a resolved _label/_en/_ar/
    # _name/_summary column (every entity table in this app produces
    # these) rather than just an "id" — a narrow `fields` request can
    # strip "id" out entirely, but a resolved column like status_label
    # survives since it's what was actually asked for. This keeps the
    # check from ever firing on homogeneous SQL aggregate rows
    # (query_education_data — year/region/totals, no such columns), which
    # already pivot correctly and generically via Case 3 below.
    is_entity_rows = any(
        isinstance(r, dict) and (
            "id" in r or any(k.endswith(("_label", "_en", "_ar", "_name", "_summary")) for k in r)
        )
        for r in tool_results
    )
    if is_entity_rows:
        metric_field = _match_field(
            query_lower, tool_results, numeric=True,
            exclude=frozenset({"allocated_budget", "spent_budget", "budget_left"}),
        )
        if metric_field:
            items = [
                (_entity_display_name(item), _to_float(item.get(metric_field)))
                for item in tool_results
            ]
            items = [(n, v) for n, v in items if n and v is not None]
            if items:
                return {
                    "labels": [n for n, _ in items],
                    "datasets": [{
                        "label": _normalize_field(metric_field).title(),
                        "data": [v for _, v in items],
                    }],
                }

        group_field = _match_field(query_lower, tool_results, numeric=False)
        if not group_field:
            # Nothing matched by FIELD name — try matching by the actual
            # VALUES instead (e.g. "completed, pending and unassigned
            # facility tasks" never says "status").
            group_field = _match_field_by_value(query_lower, tool_results)
        if group_field:
            counts: Dict[str, int] = {}
            for item in tool_results:
                val = item.get(group_field)
                if val:
                    counts[str(val)] = counts.get(str(val), 0) + 1
            # Only a genuine distribution when values repeat (e.g. many
            # meetings sharing a few venues) — if every row has its own
            # distinct value, counting isn't meaningful.
            if 2 <= len(counts) < len(tool_results):
                # Highest count first — what "top N" means, and a
                # sensible default reading order even when N isn't named.
                ranked = sorted(counts.items(), key=lambda kv: -kv[1])
                top_n = re.search(r"\btop (\d+)\b", query_lower)
                if top_n:
                    ranked = ranked[: int(top_n.group(1))]
                return {
                    "labels": [k for k, _ in ranked],
                    "datasets": [{"label": "Count", "data": [v for _, v in ranked]}],
                }

    # Case 1b: a plain count per filtered subject ("number of projects X
    # is included in", or "...X and Y...") — checked before budget below,
    # which would otherwise unconditionally win on project/office rows.
    subject_count_chart = _extract_subject_count_chart(query_lower, tool_results)
    if subject_count_chart:
        return subject_count_chart

    # Case 2: Budget comparison across multiple items (projects, SG offices)
    budget_items = []
    for item in tool_results:
        name = (
            item.get("project_name_en")
            or item.get("sg_office_name_en")
            or item.get("task_name")
            or item.get("resolution_topic_en")
        )
        if not name:
            # Check nested structures
            for key in ["project", "office", "task", "resolution"]:
                nested = item.get(key)
                if isinstance(nested, dict):
                    name = (
                        nested.get("project_name_en")
                        or nested.get("sg_office_name_en")
                        or nested.get("task_name")
                        or nested.get("resolution_topic_en")
                    )
                    break

        budget = item.get("budget")
        if isinstance(budget, dict):
            alloc = _to_float(budget.get("allocated_budget"))
            spent = _to_float(budget.get("spent_budget"))
            left = _to_float(budget.get("budget_left"))
            if name and alloc is not None:
                budget_items.append({
                    "name": name,
                    "allocated": alloc,
                    "spent": spent or 0,
                    "left": left or 0,
                })
        elif name:
            alloc = _to_float(item.get("allocated_budget"))
            spent = _to_float(item.get("spent_budget"))
            if alloc is not None:
                budget_items.append({
                    "name": name,
                    "allocated": alloc,
                    "spent": spent or 0,
                    "left": _to_float(item.get("budget_left")) or 0,
                })

    if budget_items:
        labels = [b["name"] for b in budget_items]
        # Tasks and council affairs (resolutions) have no budget field in
        # this schema at all, so if the tool data mixed them in with
        # projects/SG offices, they were silently skipped above — flag that
        # here so the caller can add a plain note instead of leaving the
        # model's text implying full coverage.
        has_no_budget_entity = any(
            (item.get("task_name") or item.get("resolution_topic_en"))
            for item in tool_results
        )
        return {
            "labels": labels,
            "datasets": [
                {"label": "Allocated", "data": [b["allocated"] for b in budget_items]},
                {"label": "Spent", "data": [b["spent"] for b in budget_items]},
                {"label": "Remaining", "data": [b["left"] for b in budget_items]},
            ],
            "no_budget_entities_present": has_no_budget_entity,
        }

    # The query explicitly asked for budget data, but no item had a usable
    # budget field (e.g. the user's role lacks Budget Info access, so
    # list_projects never attached allocated_budget/spent_budget to the
    # results). Stop here instead of falling through to the generic
    # fallbacks below — those would happily chart whatever OTHER numeric
    # field survives (status codes, manager IDs, ...) as if it were the
    # requested budget data, producing a real-looking but meaningless chart.
    if "budget" in query_lower:
        return None

    # Case 3: Homogeneous row dicts (e.g. SQL query results like
    # {"year": 2022, "total_students": 509738, "total_schools": 776}) —
    # pivot into a proper grouped-bar chart: one categorical column for the
    # x-axis, each remaining numeric column becomes its own series. This is
    # what makes "bar chart of X and Y by year" show real grouped bars
    # instead of one bar per numeric field with no shared category.
    row_dict_chart = _extract_row_dicts_chart(tool_results)
    if row_dict_chart:
        return row_dict_chart

    # Case 4: Generic numeric data extraction (fallback for anything that
    # doesn't look like a set of homogeneous rows)
    all_keys = set()
    for row in tool_results:
        if isinstance(row, dict):
            all_keys.update(row.keys())
    # Skip a raw code superseded by a sibling field that actually holds
    # real text — same rule as elsewhere, so a status/venue/etc code
    # never gets plotted as if it were a real number.
    drop_keys = _raw_code_keys(all_keys, tool_results)

    numeric_items = []
    for item in tool_results:
        name = (
            item.get("project_name_en")
            or item.get("sg_office_name_en")
            or item.get("task_name")
            or item.get("resolution_topic_en")
            or ""
        )
        for key, val in item.items():
            fval = _to_float(val)
            if (
                fval is not None and fval > 0
                and key != "id" and not key.endswith("_id")
                and key not in drop_keys
            ):
                numeric_items.append({"label": f"{name} - {key}", "value": fval})

    if len(numeric_items) >= 2:
        labels = [n["label"] for n in numeric_items[:15]]
        values = [n["value"] for n in numeric_items[:15]]
        return {
            "labels": labels,
            "datasets": [{"label": "Value", "data": values}],
            "chart_type": "bar",
        }

    # Case 5: Last resort. The query asked for *a* chart of some entity list
    # (e.g. "generate a chart of tasks", or a color-only follow-up like
    # "show in orange" that names no field at all) without naming any
    # specific numeric field, and none of the cases above found one.
    # Status is the one dimension every entity list here usually has, so
    # try that first — but a follow-up can land on data narrowed to some
    # OTHER field only (e.g. priority-only, from the prior turn's own
    # field selection), so fall back further to whichever real categorical
    # field is actually present, instead of giving up outright.
    return (
        _status_distribution_chart(tool_results)
        or _any_available_distribution_chart(tool_results)
    )


def _any_available_distribution_chart(tool_results: List[Dict]) -> Optional[Dict]:
    """Picks whichever real categorical field has the most genuine
    distribution, using the same candidate rules as _match_field but
    without requiring the query to name it — the true last resort."""
    keys = set()
    for row in tool_results:
        if isinstance(row, dict):
            keys.update(row.keys())
    keys -= _ENTITY_NAME_FIELDS
    keys = {k for k in keys if k != "id" and not k.endswith("_id")}
    keys -= _raw_code_keys(keys, tool_results)

    best_counts, best_total = None, 0
    for key in keys:
        if _field_is_numeric(key, tool_results):
            continue
        counts: Dict[str, int] = {}
        for row in tool_results:
            val = row.get(key)
            if val:
                counts[str(val)] = counts.get(str(val), 0) + 1
        total = sum(counts.values())
        if 2 <= len(counts) < len(tool_results) and total > best_total:
            best_counts, best_total = counts, total

    if not best_counts:
        return None
    return {
        "labels": list(best_counts.keys()),
        "datasets": [{"label": "Count", "data": list(best_counts.values())}],
    }


def _status_distribution_chart(tool_results: List[Dict]) -> Optional[Dict]:
    """Count items by status field and return chart_data, or None if fewer
    than 2 distinct statuses are present (nothing meaningful to compare)."""
    status_counts = {}
    for item in tool_results:
        status = (
            item.get("status_label")
            # Visitor readiness rows have no "status_label" field — their
            # distribution-worthy field is readiness_status_label instead —
            # without this, a "visitor readiness status" chart request fell
            # through to the generic numeric fallback and plotted IDs.
            or item.get("readiness_status_label")
            or item.get("status_en")
            or item.get("status")
        )
        if status:
            label = str(status)
            status_counts[label] = status_counts.get(label, 0) + 1

    if len(status_counts) < 2:
        return None

    return {
        "labels": list(status_counts.keys()),
        "datasets": [{"label": "Count", "data": list(status_counts.values())}],
    }


def _extract_row_dicts_chart(tool_results: List[Dict]) -> Optional[Dict]:
    """
    Pivot a list of homogeneous row dicts (e.g. query_education_data results,
    already expanded from {"columns","rows"} into one dict per row) into a
    grouped bar chart: pick one categorical column for the x-axis labels,
    and every numeric column becomes its own series.

    Example: [{"year": 2022, "total_students": 509738, "total_schools": 776},
              {"year": 2023, "total_students": 527759, "total_schools": 757}]
    -> labels ["2022", "2023"], datasets for "Total Students" and "Total Schools".
    """
    rows = [r for r in tool_results if isinstance(r, dict)]
    if len(rows) < 2:
        return None

    common_keys = set.intersection(*(set(r.keys()) for r in rows))
    if not common_keys:
        return None

    # Any foreign-key id (meeting_request_id, arrival_marked_by_id, ...) is
    # never a real metric, not just the 3 literal names caught before.
    EXCLUDE = {k for k in common_keys if k == "id" or k.endswith("_id")}
    # A raw code (status=2) superseded by a sibling field holding real
    # text is never a real metric — same rule _match_field uses, so this
    # pivot doesn't plot raw enum codes as if they were real numbers.
    EXCLUDE |= _raw_code_keys(common_keys, rows)

    # Prefer a conventionally-named category column for the x-axis. Entity
    # "name" fields come first — without this, the generic fallback below
    # picks whichever non-numeric column sorts alphabetically first, which
    # for tasks/offices/resolutions is usually "category_name" (a shared,
    # repeated value across rows), not the field that actually identifies
    # each individual record (e.g. task_name).
    preferred_label_cols = [
        "task_name", "project_name_en", "sg_office_name_en", "resolution_topic_en",
        "year", "label", "name", "category", "region", "period", "sector",
    ]
    label_col = next((c for c in preferred_label_cols if c in common_keys), None)

    # Otherwise pick the first column that isn't purely numeric across all rows.
    if label_col is None:
        for key in sorted(common_keys - EXCLUDE):
            if any(_to_float(r.get(key)) is None for r in rows):
                label_col = key
                break

    numeric_cols = [
        key for key in sorted(common_keys - EXCLUDE - {label_col})
        if all(_to_float(r.get(key)) is not None for r in rows)
    ]
    if not numeric_cols:
        return None

    labels = [str(r.get(label_col, f"Row {i + 1}")) for i, r in enumerate(rows)] if label_col else [
        f"Row {i + 1}" for i in range(len(rows))
    ]

    datasets = [
        {
            "label": col.replace("_", " ").title(),
            "data": [_to_float(r.get(col)) or 0 for r in rows],
        }
        for col in numeric_cols[:6]
    ]

    return {"labels": labels, "datasets": datasets, "chart_type": "bar"}


_MAGNITUDE_SUFFIXES = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
}
_MAGNITUDE_RE = re.compile(
    r"^([\d.]+)\s*(k|m|b|thousand|million|billion)?$", re.IGNORECASE
)


def _to_float(val) -> Optional[float]:
    """
    Try converting value to float. Budget fields in this DB are sometimes
    stored as human-typed strings with a magnitude suffix (e.g. "25 Million",
    "0 M") rather than plain numbers — a bare float() call on those silently
    fails and drops the project from any budget chart entirely, even though
    real budget data exists. Understands k/m/b and thousand/million/billion,
    case-insensitive, in addition to plain numbers.
    """
    if val is None:
        return None
    s = str(val).replace(",", "").strip()
    if not s:
        return None
    m = _MAGNITUDE_RE.match(s)
    if not m:
        return None
    try:
        number = float(m.group(1))
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    return number * _MAGNITUDE_SUFFIXES.get(suffix, 1)


_CHART_TYPE_WORDS = {
    "pie": ("pie", "distribution", "percentage", "share", "breakdown"),
    "line": ("line", "trend", "over time", "timeline"),
    "bar": ("bar",),
}


def _infer_chart_type(query: str, labels: list) -> str:
    """Infer appropriate chart type from query and data shape."""
    if any(w in query for w in _CHART_TYPE_WORDS["pie"]):
        return "pie"
    if any(w in query for w in _CHART_TYPE_WORDS["line"]):
        return "line"
    return "bar"


def _query_names_chart_type(query: str) -> bool:
    """True if the query itself names pie/line/bar explicitly — an
    explicit type THIS turn must always win over one blended in from the
    previous turn, never get silently overridden by stale context."""
    return any(w in query for words in _CHART_TYPE_WORDS.values() for w in words)


def _nice_integer_dtick(max_value: float) -> int:
    """A whole-number Y-axis tick interval, so count data (never
    fractional) shows gridlines like 1, 2, 3 instead of Plotly's default
    auto-ticks for a small numeric range, which land on fractions like 1,
    1.5, 2, 2.5 — not a missing label, just a confusing axis for a count
    ("2.5 emails" means nothing)."""
    if max_value <= 10:
        return 1
    raw_step = max_value / 8
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for mult in (1, 2, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            return int(step)
    return int(magnitude * 10)


def _build_plotly_config(
    chart_type: str, labels: List[str], datasets: List[Dict], palette_colors: List[str] = BROWN_COLORS
) -> Dict[str, Any]:
    """Build Plotly.js data + layout config (no hardcoded container ID)."""
    colors = palette_colors[: max(len(labels), len(datasets))]

    traces = []
    for i, ds in enumerate(datasets):
        if chart_type == "pie":
            traces.append({
                "labels": labels,
                "values": ds["data"],
                "type": "pie",
                "marker": {"colors": colors[: len(labels)]},
                "name": ds.get("label", ""),
            })
        elif chart_type == "line":
            traces.append({
                "x": labels,
                "y": ds["data"],
                "type": "scatter",
                "mode": "lines+markers",
                "name": ds.get("label", ""),
                "line": {"color": colors[i % len(colors)]},
            })
        else:  # bar
            traces.append({
                "x": labels,
                "y": ds["data"],
                "type": "bar",
                "name": ds.get("label", ""),
                "marker": {"color": colors[i % len(colors)]},
            })

    layout = {
        "title": "",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "legend": {"orientation": "h", "y": -0.2},
    }

    if chart_type == "bar":
        layout["barmode"] = "group"

    # Budget figures (Allocated/Spent/Remaining) are already formatted by
    # the frontend ("2M" etc.) — adding our own dtick/tickformat here
    # risked clashing with that, so this is scoped to non-budget (count)
    # charts only, by the one signal that reliably tells them apart: the
    # budget triple's dataset labels are always exactly these three.
    is_budget_chart = any(ds.get("label") in ("Allocated", "Spent", "Remaining") for ds in datasets)
    if chart_type in ("bar", "line") and not is_budget_chart:
        all_values = [v for ds in datasets for v in ds.get("data", []) if isinstance(v, (int, float))]
        if all_values:
            layout["yaxis"] = {
                "dtick": _nice_integer_dtick(max(all_values)),
                "rangemode": "tozero",
            }

    return {"data": traces, "layout": layout}


def validate_output_consistency(
    chart_data: Optional[Dict], tool_results: Optional[List[Dict]], assistant_text: str
) -> List[str]:
    """Deterministic, log-only sanity checks comparing a chart's actual
    numbers against the raw rows they were built from, and against the
    text describing them. Returns warning strings for the caller to log —
    never blocks or rewrites the response, since a false positive here
    (e.g. the model paraphrased "no projects" instead of writing "0")
    should never break a legitimate answer. This is observability, not a
    gate: exactly the kind of check that would have caught the duplicate-
    tool-call chart-doubling bug (a true count of 3 rendered as 6)
    automatically instead of needing a human to spot a bad screenshot.

    Reads the actual top-level shape detect_chart_opportunity returns —
    chart_data["plotly_data"], a list of Plotly trace dicts (x/y for
    bar/line, labels/values for pie, "name" for the series label) — not
    the internal {"labels":[...], "datasets":[...]} shape some helpers
    inside this module use before it gets converted; those never reach
    the caller directly.
    """
    warnings: List[str] = []
    if not chart_data or not isinstance(tool_results, list):
        return warnings

    row_count = len(tool_results)
    text = assistant_text or ""

    for trace in chart_data.get("plotly_data") or []:
        trace_labels = trace.get("x") or trace.get("labels") or []
        values = trace.get("y") or trace.get("values") or []
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            continue
        # Only meaningful for literal per-row COUNTS — budgets/amounts are
        # a different unit entirely; summing them against a row count
        # would never mean anything. The trace's own name is how we know
        # which kind this is (see _extract_subject_count_chart / the
        # Case 1 distribution path in _extract_chart_data).
        if str(trace.get("name", "")).strip().lower() != "count":
            continue

        total = sum(numeric_values)
        if total > row_count:
            warnings.append(
                f"chart trace 'Count' sums to {total}, which EXCEEDS the "
                f"{row_count} rows actually fetched this turn (labels={trace_labels}) "
                f"— likely duplicate-counted data."
            )

        for label, val in zip(trace_labels, numeric_values):
            if str(int(val)) not in text:
                warnings.append(
                    f"chart shows {label!r} = {val!r} but that number does not "
                    f"appear anywhere in the response text — possible text/chart mismatch."
                )

    return warnings
