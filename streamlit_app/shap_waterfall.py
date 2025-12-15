"""
Custom SHAP waterfall plot for Streamlit.

Features:
- Waterfall-style contribution bars
- Feature labels with type indicators (R/D/A) and missing marker [M]
- Right-aligned column for feature values (raw or derived)
- Matplotlib backend for Streamlit compatibility
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def _format_value(value: object) -> str:
    """Format feature values for display in the right column."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        # Use compact formatting while keeping readability
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def _build_labels(
    feature_names: Iterable[str],
    feature_values: Dict[str, object],
    feature_metadata: Dict[str, Dict[str, object]] | None,
) -> Tuple[list[str], list[str]]:
    """Return formatted labels and value strings aligned by row."""
    labels = []
    values = []

    for name in feature_names:
        meta = feature_metadata.get(name, {}) if feature_metadata else {}
        feat_type = meta.get("type", "R")
        missing_flag = meta.get("missing")
        value = feature_values.get(name)

        # Determine missing if not explicitly provided
        if missing_flag is None:
            missing_flag = value is None or (isinstance(value, float) and np.isnan(value))

        missing_prefix = "[M] " if missing_flag else ""
        label = f"{missing_prefix}{name} ({feat_type})"
        labels.append(label)
        values.append(_format_value(value))

    return labels, values


def plot_shap_waterfall(
    shap_values: Dict[str, float],
    feature_values: Dict[str, object],
    feature_metadata: Dict[str, Dict[str, object]] | None = None,
    *,
    base_value: float | None = None,
    prediction_value: float | None = None,
    max_display: int = 15,
    figsize: Tuple[int, int] | None = None,
):
    """Create a custom SHAP waterfall-style plot.

    Args:
        shap_values: Mapping of feature -> shap contribution.
        feature_values: Mapping of feature -> feature value (raw or derived).
        feature_metadata: Optional mapping with keys: type (R/D/A), missing (bool).
        base_value: Optional model base value for reference line.
        prediction_value: Optional predicted score to annotate.
        max_display: Maximum number of features to display (sorted by |SHAP|).
        figsize: Optional figure size; if None, size is based on number of features.

    Returns:
        matplotlib Figure ready for Streamlit display (st.pyplot).
    """

    if not shap_values:
        raise ValueError("No SHAP values provided for waterfall plot")

    # Sort features by absolute SHAP contribution
    series = pd.Series(shap_values, dtype=float)
    series = series.reindex(series.abs().sort_values(ascending=False).index)
    series = series.head(max_display)

    n_features = len(series)
    if figsize is None:
        # Height scales with features; width leaves room for value column
        figsize = (10, max(4, int(n_features * 0.5)))

    labels, value_strings = _build_labels(series.index, feature_values, feature_metadata)

    # Setup figure with two axes: bars + value column
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_vals = fig.add_subplot(gs[1], sharey=ax)

    y_positions = np.arange(n_features)

    # Draw horizontal waterfall bars
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in series]
    ax.barh(y_positions, series.values, color=colors, alpha=0.9)

    # Vertical reference lines
    ax.axvline(0, color="#666", linewidth=1, linestyle="--", alpha=0.6)
    if base_value is not None:
        ax.axvline(base_value, color="#1f77b4", linewidth=1, linestyle=":", alpha=0.8)
    if prediction_value is not None:
        ax.axvline(prediction_value, color="#9467bd", linewidth=1, linestyle="-", alpha=0.6)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Most important at top

    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("Feature")

    # Remove spines for cleaner look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Right column for feature values
    ax_vals.set_xlim(0, 1)
    ax_vals.set_ylim(ax.get_ylim())
    ax_vals.set_xticks([])
    ax_vals.set_yticks([])
    ax_vals.spines["top"].set_visible(False)
    ax_vals.spines["right"].set_visible(False)
    ax_vals.spines["left"].set_visible(False)
    ax_vals.spines["bottom"].set_visible(False)

    for y, val_str in zip(y_positions, value_strings):
        ax_vals.text(0.02, y, val_str, va="center", ha="left", fontsize=10)

    # Title / annotations
    title_parts = ["SHAP Waterfall"]
    if prediction_value is not None:
        title_parts.append(f"Prediction: {prediction_value:.3f}")
    if base_value is not None:
        title_parts.append(f"Base: {base_value:.3f}")
    ax.set_title(" | ".join(title_parts), loc="left", fontsize=12, pad=12)

    plt.tight_layout()
    return fig
