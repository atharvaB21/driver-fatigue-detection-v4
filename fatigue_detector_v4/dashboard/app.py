"""Streamlit dashboard for fatigue detection session analysis.

Connects to the same SQLite database used by the fatigue detector
and provides interactive visualizations of EAR, MAR, PERCLOS,
head pose angles, gaze tracking, distraction breakdowns,
state transitions, and session summaries.

Run with: streamlit run dashboard/app.py
"""

import os
import sys

import streamlit as st
import sqlite3
import pandas as pd

# Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Default DB path (sibling of dashboard/ directory)
DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'fatigue_log.db'
)

# New columns introduced in the production schema.  Older databases
# may not contain them, so we detect their presence at runtime.
_NEW_COLUMNS = {'distraction_type', 'gaze_h', 'gaze_v', 'face_confidence'}


def _table_columns(db_path: str, table: str = 'events') -> set[str]:
    """Return the set of column names present in *table*."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


def load_data(db_path: str) -> pd.DataFrame:
    """Load event data from SQLite database into a DataFrame.

    Gracefully handles databases that lack the newer columns
    (``distraction_type``, ``gaze_h``, ``gaze_v``, ``face_confidence``)
    by filling them with sensible defaults after loading.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query('SELECT * FROM events ORDER BY id ASC', conn)
    finally:
        conn.close()

    if df.empty:
        return df

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Back-fill missing production columns so downstream code never
    # has to guard against KeyError.
    if 'distraction_type' not in df.columns:
        df['distraction_type'] = 'NONE'
    if 'gaze_h' not in df.columns:
        df['gaze_h'] = 0.0
    if 'gaze_v' not in df.columns:
        df['gaze_v'] = 0.0
    if 'face_confidence' not in df.columns:
        df['face_confidence'] = None

    return df


# ────────────────────────────────────────────────────────────────────
# Chart helpers
# ────────────────────────────────────────────────────────────────────

def _plot_ear_mar_timeline(df: pd.DataFrame) -> None:
    """Render EAR & MAR over time as an interactive line chart."""
    st.subheader("📈 EAR & MAR Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ear'],
        name='EAR', line=dict(color='cyan', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['mar'],
        name='MAR', line=dict(color='orange', width=1.5)
    ))
    fig.add_hline(y=0.25, line_dash='dash', line_color='red',
                  annotation_text='EAR Threshold (0.25)')
    fig.add_hline(y=0.60, line_dash='dash', line_color='yellow',
                  annotation_text='MAR Threshold (0.60)')
    fig.update_layout(
        template='plotly_dark', height=400,
        xaxis_title='Time', yaxis_title='Ratio',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_perclos_trend(df: pd.DataFrame) -> None:
    """Render PERCLOS trend line with fatigue threshold."""
    st.subheader("📉 PERCLOS Trend")
    fig = px.line(df, x='timestamp', y='perclos', template='plotly_dark')
    fig.add_hline(y=0.15, line_dash='dash', line_color='red',
                  annotation_text='Fatigue Threshold (0.15)')
    fig.update_layout(height=300, xaxis_title='Time', yaxis_title='PERCLOS')
    st.plotly_chart(fig, use_container_width=True)


def _plot_head_pose(df: pd.DataFrame) -> None:
    """Render pitch and yaw head-pose charts side by side."""
    st.subheader("🔄 Head Pose Angles")
    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.line(df, x='timestamp', y='pitch',
                      title='Pitch (Nodding)', template='plotly_dark')
        fig.add_hline(y=20, line_dash='dash', line_color='orange',
                      annotation_text='+20°')
        fig.add_hline(y=-20, line_dash='dash', line_color='orange',
                      annotation_text='-20°')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.line(df, x='timestamp', y='yaw',
                      title='Yaw (Looking Away)', template='plotly_dark')
        fig.add_hline(y=30, line_dash='dash', line_color='orange',
                      annotation_text='+30°')
        fig.add_hline(y=-30, line_dash='dash', line_color='orange',
                      annotation_text='-30°')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def _plot_distraction_breakdown(df: pd.DataFrame) -> None:
    """Pie chart showing distribution of distraction types."""
    st.subheader("🥧 Distraction Breakdown")

    distraction_df = df[
        (df['distraction_type'] != 'NONE')
        & (df['event_type'] == 'DISTRACTION')
    ]

    if distraction_df.empty:
        st.info("No distraction events recorded yet.")
        return

    counts = (
        distraction_df['distraction_type']
        .value_counts()
        .reset_index()
    )
    counts.columns = ['Distraction Type', 'Count']

    fig = px.pie(
        counts,
        names='Distraction Type',
        values='Count',
        template='plotly_dark',
        hole=0.35,
        title='Distraction Type Distribution',
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def _plot_alert_timeline(df: pd.DataFrame) -> None:
    """Scatter / timeline plot of DISTRACTION events coloured by type."""
    st.subheader("⏱️ Alert Timeline")

    alert_df = df[df['event_type'] == 'DISTRACTION'].copy()

    if alert_df.empty:
        st.info("No distraction alerts recorded yet.")
        return

    fig = px.scatter(
        alert_df,
        x='timestamp',
        y='distraction_type',
        color='distraction_type',
        template='plotly_dark',
        title='Distraction Events Over Time',
        labels={
            'timestamp': 'Time',
            'distraction_type': 'Distraction Type',
        },
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def _plot_gaze_heatmap(df: pd.DataFrame) -> None:
    """2-D density heatmap of gaze coordinates from SNAPSHOT events."""
    st.subheader("👁️ Gaze Heatmap")

    snapshot_df = df[df['event_type'] == 'SNAPSHOT'].copy()

    # Filter out rows where gaze data is missing / all-zero
    snapshot_df = snapshot_df[
        (snapshot_df['gaze_h'].notna())
        & (snapshot_df['gaze_v'].notna())
        & ~((snapshot_df['gaze_h'] == 0.0) & (snapshot_df['gaze_v'] == 0.0))
    ]

    if snapshot_df.empty:
        st.info("No gaze data available in SNAPSHOT events.")
        return

    fig = px.density_heatmap(
        snapshot_df,
        x='gaze_h',
        y='gaze_v',
        template='plotly_dark',
        title='Gaze Position Density',
        labels={
            'gaze_h': 'Horizontal Gaze (Left → Right)',
            'gaze_v': 'Vertical Gaze (Up → Down)',
        },
        color_continuous_scale='Inferno',
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="Driver Fatigue Monitor",
        layout="wide",
        page_icon="🚗"
    )

    st.title("🚗 Driver Fatigue Monitor — Session Dashboard")

    # ── Sidebar Controls ──
    db_path = st.sidebar.text_input("Database Path", DEFAULT_DB)

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh (5s)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Usage:** Run the fatigue detector first to generate data, "
        "then open this dashboard to visualize the session."
    )

    # ── Load Data ──
    df = load_data(db_path)

    if df.empty:
        st.warning(
            "⚠️ No data found. Run the fatigue detector first to "
            "generate session data, then refresh this dashboard."
        )
        return

    # ── Summary Cards (8 metrics) ──
    st.subheader("📊 Session Summary")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)

    state_changes = df[df['event_type'] == 'STATE_CHANGE']

    c1.metric("Total Events", len(df))
    c2.metric("Drowsy Alerts",
              len(state_changes[state_changes['state'] == 'DROWSY']))
    c3.metric("Very Drowsy",
              len(state_changes[state_changes['state'] == 'VERY_DROWSY']))
    c4.metric("Asleep Alerts",
              len(state_changes[state_changes['state'] == 'ASLEEP']))
    c5.metric("Distractions", int(df['distracted'].sum()) if 'distracted' in df.columns else 0)
    c6.metric("Yawns", int(df['yawning'].sum()) if 'yawning' in df.columns else 0)
    c7.metric("No Face Events",
              int((df['distraction_type'] == 'NO_FACE').sum()))
    c8.metric("RTSP Issues",
              int(df['event_type'].str.startswith('RTSP_').sum()))

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    st.markdown("---")

    # ── EAR & MAR Timeline ──
    _plot_ear_mar_timeline(df)

    # ── PERCLOS Trend ──
    _plot_perclos_trend(df)

    # ── Head Pose ──
    _plot_head_pose(df)

    st.markdown("---")

    # ── Distraction Breakdown (pie) & Alert Timeline (scatter) ──
    col_pie, col_timeline = st.columns(2)
    with col_pie:
        _plot_distraction_breakdown(df)
    with col_timeline:
        _plot_alert_timeline(df)

    st.markdown("---")

    # ── Gaze Heatmap ──
    _plot_gaze_heatmap(df)

    st.markdown("---")

    # ── State Changes Table ──
    st.subheader("📋 State Change Log (Latest First)")
    if not state_changes.empty:
        display_cols = [
            'timestamp', 'state', 'ear', 'mar',
            'perclos', 'pitch', 'yaw',
            'gaze_h', 'gaze_v', 'distraction_type',
        ]
        # Only keep columns that actually exist in the dataframe
        display_cols = [c for c in display_cols if c in state_changes.columns]
        st.dataframe(
            state_changes[display_cols].iloc[::-1].head(50),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No state changes recorded yet.")

    # ── Raw Data ──
    with st.expander("🗄️ View Raw Data (Latest First)"):
        st.dataframe(
            df.iloc[::-1].head(200),
            use_container_width=True,
            hide_index=True,
        )

    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == '__main__':
    main()
