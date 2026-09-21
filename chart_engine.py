"""
Chart detection and visualization JSON generation for EHCD Chatbot.
Inspects tool results and user query to determine if a chart/graph is appropriate,
then generates Chart.js-compatible JSON configuration.
"""

from typing import Any, Dict, List, Optional

# Brown color palette as specified
BROWN_COLORS = [
    "#8B4513",  # SaddleBrown
    "#A0522D",  # Sienna
    "#CD853F",  # Peru
    "#DEB887",  # BurlyWood
    "#D2691E",  # Chocolate
    "#BC8F8F",  # RosyBrown
    "#F4A460",  # SandyBrown
    "#DAA520",  # GoldenRod
    "#B8860B",  # DarkGoldenRod
    "#D2B48C",  # Tan
]


def detect_chart_opportunity(
    query: str,
    tool_results: List[Dict],
    assistant_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Analyze query + tool results to determine if a chart is appropriate.
    Returns chart_data dict or None.
    """
    query_lower = query.lower()

    # Check if user explicitly asks for chart/graph
    explicit_chart = any(
        w in query_lower
        for w in ["chart", "graph", "visualiz", "diagram", "compare budget", "pie", "bar chart"]
    )

    if not explicit_chart:
        return None

    # Try to extract chartable data from tool results
    chart_data = _extract_chart_data(query_lower, tool_results)
    if not chart_data:
        return None

    labels = chart_data["labels"]
    datasets = chart_data["datasets"]
    chart_type = chart_data.get("chart_type", _infer_chart_type(query_lower, labels))

    if len(labels) < 1:
        return None

    # Generate Plotly config (data + layout, no hardcoded container ID)
    plotly_config = _build_plotly_config(chart_type, labels, datasets)

    return {
        "chart_type": chart_type,
        "plotly_data": plotly_config["data"],
        "plotly_layout": plotly_config["layout"],
        "audit": {
            "total_items": len(labels),
            "labels": labels[:20],
        },
    }


def _extract_chart_data(
    query_lower: str, tool_results: List[Dict]
) -> Optional[Dict]:
    """Extract label-value pairs from tool results for charting."""

    if not tool_results:
        return None

    # Case 1: Budget comparison across multiple items (projects, SG offices)
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
        return {
            "labels": labels,
            "datasets": [
                {"label": "Allocated", "data": [b["allocated"] for b in budget_items]},
                {"label": "Spent", "data": [b["spent"] for b in budget_items]},
                {"label": "Remaining", "data": [b["left"] for b in budget_items]},
            ],
            "chart_type": "bar",
        }

    # Case 2: Status distribution
    if any(w in query_lower for w in ["status", "distribution", "breakdown", "pie"]):
        status_counts = {}
        for item in tool_results:
            status = (
                item.get("status_label")
                or item.get("status_en")
                or item.get("status")
            )
            if status:
                label = str(status)
                status_counts[label] = status_counts.get(label, 0) + 1

        if len(status_counts) >= 2:
            labels = list(status_counts.keys())
            values = list(status_counts.values())
            return {
                "labels": labels,
                "datasets": [{"label": "Count", "data": values}],
                "chart_type": "pie",
            }

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
            if fval is not None and fval > 0 and key not in ("id", "task_id", "resolution_id"):
                numeric_items.append({"label": f"{name} - {key}", "value": fval})

    if len(numeric_items) >= 2:
        labels = [n["label"] for n in numeric_items[:15]]
        values = [n["value"] for n in numeric_items[:15]]
        return {
            "labels": labels,
            "datasets": [{"label": "Value", "data": values}],
            "chart_type": "bar",
        }

    return None


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

    EXCLUDE = {"id", "task_id", "resolution_id"}

    # Prefer a conventionally-named category column for the x-axis.
    preferred_label_cols = ["year", "label", "name", "category", "region", "period", "sector"]
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


def _to_float(val) -> Optional[float]:
    """Try converting value to float."""
    if val is None:
        return None
    try:
        s = str(val).replace(",", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


def _infer_chart_type(query: str, labels: list) -> str:
    """Infer appropriate chart type from query and data shape."""
    if any(w in query for w in ["pie", "distribution", "percentage", "share", "breakdown"]):
        return "pie"
    if any(w in query for w in ["line", "trend", "over time", "timeline"]):
        return "line"
    return "bar"


def _build_plotly_config(
    chart_type: str, labels: List[str], datasets: List[Dict]
) -> Dict[str, Any]:
    """Build Plotly.js data + layout config (no hardcoded container ID)."""
    colors = BROWN_COLORS[: max(len(labels), len(datasets))]

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

    return {"data": traces, "layout": layout}
