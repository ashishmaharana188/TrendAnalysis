from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def synchronized_series_chart(
    series: Mapping[str, pd.DataFrame],
    start_date: date,
    end_date: date,
):
    nonempty: list[tuple[str, pd.DataFrame]] = []
    for label, frame in series.items():
        if frame is None or frame.empty:
            continue
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))]
        if not out.empty:
            nonempty.append((label, out))
    if not nonempty:
        return None

    fig = make_subplots(
        rows=len(nonempty), cols=1, shared_xaxes=True,
        vertical_spacing=min(0.08, 0.25 / max(1, len(nonempty))),
        subplot_titles=[label for label, _ in nonempty],
    )
    for idx, (label, frame) in enumerate(nonempty, start=1):
        fig.add_trace(
            go.Scatter(
                x=frame["date"], y=frame["value"], name=label,
                mode="lines", line={"width": 1.8},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.5g}<extra>" + label + "</extra>",
            ),
            row=idx, col=1,
        )
        fig.update_yaxes(title_text="Value", row=idx, col=1, fixedrange=False)
    fig.update_layout(
        height=max(340, len(nonempty) * 230),
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        showlegend=False,
    )
    fig.update_xaxes(matches="x", showspikes=True, spikemode="across", spikesnap="cursor")
    return fig
