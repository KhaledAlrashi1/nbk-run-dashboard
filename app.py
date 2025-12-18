import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Reduce noisy warnings in Streamlit logs (optional)
# =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =========================
# Page config + Bright theme
# =========================
st.set_page_config(
    page_title="NBK Run Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background: #f6f9ff; }

h1, h2, h3, h4, h5, h6, p, li, div { color: #0f172a; }

.kpi-card{
  background: #ffffff;
  border: 1px solid #e6eefc;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
}
.kpi-label{ font-size: 0.85rem; color: #334155; }
.kpi-value{ font-size: 1.85rem; font-weight: 780; margin-top: 2px; }
.kpi-sub{ font-size: 0.85rem; color: #06b6d4; margin-top: 6px; }
.kpi-sub-muted{ font-size: 0.80rem; color: #64748b; margin-top: 2px; }

:root{ --primary-color: #06b6d4; }

.modebar{ opacity: 0.18 !important; }
.modebar:hover{ opacity: 1 !important; }
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"

# Blue + Green
GENDER_COLORS = {"Female": "#2563eb", "Male": "#22c55e"}


def apply_modern_hover(fig):
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#e2e8f0",
            font=dict(size=13, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"),
            align="left",
        ),
        hovermode="closest",
    )
    fig.update_traces(hoverlabel=dict(namelength=-1))
    return fig


def fix_year_axis(fig, years_sorted):
    years_sorted = [int(y) for y in years_sorted if pd.notna(y)]
    fig.update_xaxes(tickmode="array", tickvals=years_sorted, ticktext=[str(y) for y in years_sorted])
    return fig


# =========================
# Paths + data loading
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "nbk_run_all_years.csv"


@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def parse_time_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    s = "".join(ch for ch in s if (ch.isdigit() or ch == ":"))
    if not s:
        return np.nan

    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = (int(p) for p in parts)
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = (int(p) for p in parts)
            return m * 60 + sec
    except Exception:
        return np.nan
    return np.nan


def seconds_to_hhmmss(sec: float) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return "—"
    sec = int(round(float(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def contest_from_distance(d) -> str:
    try:
        v = float(d)
    except Exception:
        return "UNKNOWN"
    if math.isclose(v, 2.5):
        return "2.5K"
    if math.isclose(v, 5.0):
        return "5K"
    if math.isclose(v, 10.0):
        return "10K"
    if math.isclose(v, 11.0):
        return "11K"
    if math.isclose(v, 21.0):
        return "21K"
    return f"{v:g}K"


@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = [
        "event_year",
        "distance_km",
        "contest",
        "disability",
        "gender",
        "division",
        "overall_rank",
        "gender_rank",
        "bib",
        "name",
        "net_time",
        "gun_time",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce").astype("Int64")
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")

    # contest normalize (no "nan" string hacks)
    df["contest"] = df["contest"].astype("string").fillna("").str.strip()
    m = df["contest"].eq("")
    df.loc[m, "contest"] = df.loc[m, "distance_km"].map(contest_from_distance)

    df["contest"] = (
        df["contest"]
        .astype("string")
        .fillna("")
        .str.replace("KM", "K", case=False, regex=False)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )

    # disability normalize
    df["disability"] = df["disability"].astype("string").fillna("").str.strip()
    df.loc[df["disability"].eq(""), "disability"] = "None"

    # gender normalize
    df["gender"] = df["gender"].astype("string").fillna("").str.strip()
    df["gender"] = df["gender"].replace({"F": "Female", "M": "Male", "female": "Female", "male": "Male"})
    df["gender"] = df["gender"].astype("string").fillna("").str.title()

    # division normalize
    df["division"] = df["division"].astype("string").fillna("").str.strip()
    df.loc[df["division"].eq(""), "division"] = "Open"

    # times
    df["net_seconds"] = df["net_time"].apply(parse_time_to_seconds)
    df["gun_seconds"] = df["gun_time"].apply(parse_time_to_seconds)

    # categorical ordering
    order = pd.CategoricalDtype(["2.5K", "5K", "10K", "11K", "21K"], ordered=True)
    df["contest"] = df["contest"].astype(order)

    df["bib"] = df["bib"].astype("string").str.strip()
    df["name"] = df["name"].astype("string").str.strip()

    return df


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def fmt_pp(pp: float) -> str:
    if pp is None or (isinstance(pp, float) and np.isnan(pp)):
        return "—"
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.1f} pp"


def dedupe_finishers(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    if x["bib"].notna().any():
        x = x.dropna(subset=["bib"])
        x = x.drop_duplicates(subset=["event_year", "contest", "division", "bib"], keep="first")
    return x


def year_snapshot(fin: pd.DataFrame, year: int) -> dict:
    y = fin[fin["event_year"].eq(year)].copy()
    participants = len(y)
    if participants == 0:
        return dict(participants=0, pct_f=np.nan, pct_m=np.nan, med_net=np.nan)

    pct_f = y["gender"].eq("Female").mean()
    pct_m = y["gender"].eq("Male").mean()

    med_net = np.nan
    if y["net_seconds"].notna().any():
        med_net = float(np.nanmedian(y["net_seconds"].values))

    return dict(participants=participants, pct_f=pct_f, pct_m=pct_m, med_net=med_net)


def kpi_card(label, value, sub="", sub_muted=""):
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
  <div class="kpi-sub-muted">{sub_muted}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def bucket_open_contest(series: pd.Series, keep_11k: bool) -> pd.Series:
    s = series.astype("string").fillna("").astype(str)
    if keep_11k:
        return s
    return s.replace({"11K": "10K"})


# =========================
# Load
# =========================
if not DEFAULT_CSV.exists():
    st.error(
        "Could not load data.\n\n"
        f"Expected file at:\n{DEFAULT_CSV}\n\n"
        "Fix: ensure your CSV is committed at `data/nbk_run_all_years.csv`."
    )
    st.stop()

df = prepare(load_csv(DEFAULT_CSV))

# =========================
# Header
# =========================
st.title("NBK Run Dashboard")
st.caption("For the last four annual runs.")

years_all = sorted([int(y) for y in df["event_year"].dropna().unique().tolist()])
y_latest = max(years_all) if years_all else None
y_prev = years_all[-2] if len(years_all) >= 2 else None

# =========================
# Split: Open / Para
# =========================
OPEN_CONTESTS_RAW = ["5K", "10K", "11K", "21K"]
OPEN_CONTESTS_BUCKETED = ["5K", "10K", "21K"]

open_fin = dedupe_finishers(
    df[df["division"].ne("Para - Athletes")]
    .copy()
    .loc[lambda x: x["contest"].astype(str).isin(OPEN_CONTESTS_RAW)]
)

para_fin = dedupe_finishers(
    df[df["division"].eq("Para - Athletes")]
    .copy()
    .loc[lambda x: x["contest"].astype(str).eq("2.5K")]
)

tab_open, tab_para = st.tabs(["Open athletes", "Para-athletes"])

# =========================
# OPEN TAB
# =========================
with tab_open:
    st.subheader("Open athletes")

    if y_latest is None or y_prev is None:
        st.info("Not enough yearly data to compute year-over-year KPIs.")
    else:
        s_latest = year_snapshot(open_fin, y_latest)
        s_prev = year_snapshot(open_fin, y_prev)

        d_part = s_latest["participants"] - s_prev["participants"]
        d_pf = (s_latest["pct_f"] - s_prev["pct_f"]) * 100 if (not np.isnan(s_latest["pct_f"]) and not np.isnan(s_prev["pct_f"])) else np.nan
        d_pm = (s_latest["pct_m"] - s_prev["pct_m"]) * 100 if (not np.isnan(s_latest["pct_m"]) and not np.isnan(s_prev["pct_m"])) else np.nan
        d_med = s_latest["med_net"] - s_prev["med_net"] if (not np.isnan(s_latest["med_net"]) and not np.isnan(s_prev["med_net"])) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card(f"Participants ({y_latest})", fmt_int(s_latest["participants"]), sub=f"Δ {fmt_int(d_part)} vs {y_prev}")
        with c2:
            val = f"{(s_latest['pct_f']*100):.1f}%" if not np.isnan(s_latest["pct_f"]) else "—"
            kpi_card(f"% Female ({y_latest})", val, sub=f"{fmt_pp(d_pf)} vs {y_prev}")
        with c3:
            val = f"{(s_latest['pct_m']*100):.1f}%" if not np.isnan(s_latest["pct_m"]) else "—"
            kpi_card(f"% Male ({y_latest})", val, sub=f"{fmt_pp(d_pm)} vs {y_prev}")
        with c4:
            kpi_card(
                f"Typical finish time ({y_latest})",
                seconds_to_hhmmss(s_latest["med_net"]),
                sub=(f"Δ {seconds_to_hhmmss(d_med)} vs {y_prev}" if not np.isnan(d_med) else ""),
                sub_muted="(median net time across Open contests)",
            )

    st.divider()

    st.subheader("At a glance")

    year_opts = [y_latest] + [y for y in years_all if y != y_latest]
    at_year = st.selectbox(
        "Show distributions for",
        options=(["All years"] + year_opts),
        index=1,
        help="Defaults to the most recent year so the distribution reflects 'current' participation.",
    )

    if at_year == "All years":
        af = open_fin.copy()
        at_year_label = "All years"
        keep_11k_here = False
    else:
        at_year_int = int(at_year)
        af = open_fin[open_fin["event_year"].eq(at_year_int)].copy()
        at_year_label = str(at_year_int)
        keep_11k_here = (at_year_int == 2021)

    af = af.assign(Contest=bucket_open_contest(af["contest"], keep_11k=keep_11k_here))

    contest_order_here = ["5K", "10K", "11K", "21K"] if keep_11k_here else OPEN_CONTESTS_BUCKETED

    left, right = st.columns([1.05, 1.25])

    share = (
        af.groupby("Contest", observed=True)
        .size()
        .reset_index(name="Participants")
        .query("Participants > 0")
    )
    fig_donut = px.pie(share, names="Contest", values="Participants", hole=0.55, template=PLOTLY_TEMPLATE)
    fig_donut.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Participants: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    fig_donut.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Contest")
    apply_modern_hover(fig_donut)

    gbar = (
        af.groupby(["Contest", "gender"], observed=True)
        .size()
        .reset_index(name="Participants")
    )
    fig_bar = px.bar(
        gbar,
        x="Contest",
        y="Participants",
        color="gender",
        barmode="group",
        color_discrete_map=GENDER_COLORS,
        template=PLOTLY_TEMPLATE,
        category_orders={"Contest": contest_order_here},
    )
    fig_bar.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Contest: %{x}<br>Participants: %{y:,}<extra></extra>"
    )
    fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Contest", yaxis_title="Participants")
    apply_modern_hover(fig_bar)

    with left:
        st.plotly_chart(fig_donut, width="stretch")
        st.caption(f"Contest mix for **{at_year_label}** (Open athletes).")
    with right:
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(f"Open participation by contest in **{at_year_label}**, split by gender.")

    st.info("Context: **21K** was introduced in **2025** (so it won’t appear in earlier-year distributions).")
    if keep_11k_here:
        st.info("Context: **11K** appears in **2021**. In most charts it is grouped into **10K** for easier comparison.")
    else:
        st.info("Context: **11K** (seen in 2021) is grouped into **10K** in this dashboard for clearer analysis.")

    st.divider()

    st.subheader("Participation over time (Open)")

    tmp = open_fin.assign(contest_bucket=bucket_open_contest(open_fin["contest"], keep_11k=False))

    trend = (
        tmp.groupby(["event_year", "contest_bucket"], observed=True)
        .size()
        .reset_index(name="Participants")
        .sort_values(["event_year", "contest_bucket"])
    )

    fig_trend = px.line(
        trend,
        x="event_year",
        y="Participants",
        color="contest_bucket",
        markers=True,
        template=PLOTLY_TEMPLATE,
        category_orders={"contest_bucket": OPEN_CONTESTS_BUCKETED},
    )
    fig_trend.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Participants: %{y:,}<extra></extra>"
    )
    fig_trend.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Year",
        yaxis_title="Participants",
        legend_title_text="Contest (11K → 10K)",
    )
    fix_year_axis(fig_trend, sorted(trend["event_year"].dropna().unique().tolist()))
    apply_modern_hover(fig_trend)
    st.plotly_chart(fig_trend, width="stretch")

    st.info("Context: **11K** appears only in **2021** and is grouped into **10K** here. **21K** appears only in **2025**.")

    st.divider()

    st.subheader("Gender balance over time (Open)")

    g = (
        open_fin.groupby(["event_year", "gender"], observed=True)
        .size()
        .reset_index(name="count")
    )
    pivot = (
        g.pivot_table(index="event_year", columns="gender", values="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "Female" not in pivot.columns:
        pivot["Female"] = 0
    if "Male" not in pivot.columns:
        pivot["Male"] = 0
    pivot["Total"] = pivot["Female"] + pivot["Male"]
    pivot["Female_pct"] = np.where(pivot["Total"] > 0, pivot["Female"] / pivot["Total"], np.nan)
    pivot["Male_pct"] = np.where(pivot["Total"] > 0, pivot["Male"] / pivot["Total"], np.nan)

    years_g = sorted([int(y) for y in pivot["event_year"].dropna().unique().tolist()])

    fig_counts = go.Figure()
    fig_counts.add_trace(
        go.Bar(
            x=pivot["event_year"],
            y=pivot["Female"],
            name="Female",
            marker_color=GENDER_COLORS["Female"],
            customdata=np.stack([pivot["Female_pct"] * 100], axis=-1),
            hovertemplate="<b>Female</b><br>Year: %{x}<br>Participants: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig_counts.add_trace(
        go.Bar(
            x=pivot["event_year"],
            y=pivot["Male"],
            name="Male",
            marker_color=GENDER_COLORS["Male"],
            customdata=np.stack([pivot["Male_pct"] * 100], axis=-1),
            hovertemplate="<b>Male</b><br>Year: %{x}<br>Participants: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig_counts.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Year",
        yaxis_title="Participants",
        legend_title_text="Gender",
    )
    fix_year_axis(fig_counts, years_g)
    apply_modern_hover(fig_counts)
    st.plotly_chart(fig_counts, width="stretch")

    fig_share = go.Figure()
    fig_share.add_trace(
        go.Scatter(
            x=pivot["event_year"],
            y=pivot["Female_pct"] * 100,
            mode="lines+markers",
            name="% Female",
            hovertemplate="<b>% Female</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>",
        )
    )
    fig_share.add_trace(
        go.Scatter(
            x=pivot["event_year"],
            y=pivot["Male_pct"] * 100,
            mode="lines+markers",
            name="% Male",
            hovertemplate="<b>% Male</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>",
        )
    )
    fig_share.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Year",
        yaxis_title="Share (%)",
        legend_title_text="",
    )
    fix_year_axis(fig_share, years_g)
    apply_modern_hover(fig_share)
    st.plotly_chart(fig_share, width="stretch")

    st.divider()

    st.subheader("Finish times")
    st.info("Context: **11K** results are grouped into **10K** in finish-time charts (for a standard distance comparison).")

    perf = open_fin[open_fin["net_seconds"].notna()].copy()
    if perf.empty:
        st.info("No finish-time data available.")
    else:
        perf = perf.assign(Contest=bucket_open_contest(perf["contest"], keep_11k=False))

        med_by_contest = (
            perf.groupby("Contest", observed=True)["net_seconds"]
            .median()
            .reset_index(name="median_net_seconds")
        )
        med_by_contest["median_hhmmss"] = med_by_contest["median_net_seconds"].apply(seconds_to_hhmmss)

        fig_med = px.bar(
            med_by_contest,
            x="Contest",
            y="median_net_seconds",
            template=PLOTLY_TEMPLATE,
            category_orders={"Contest": OPEN_CONTESTS_BUCKETED},
        )
        fig_med.update_traces(
            customdata=np.stack([med_by_contest["median_hhmmss"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Typical finish time: %{customdata[0]}<extra></extra>",
        )
        fig_med.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Contest (ordered by distance)",
            yaxis_title="Median finish time (seconds)",
        )
        apply_modern_hover(fig_med)
        st.plotly_chart(fig_med, width="stretch")

        pick_default = "10K" if "10K" in OPEN_CONTESTS_BUCKETED else OPEN_CONTESTS_BUCKETED[0]
        pick = st.selectbox(
            "Pick a contest to see its trend",
            options=OPEN_CONTESTS_BUCKETED,
            index=OPEN_CONTESTS_BUCKETED.index(pick_default),
        )

        tt = perf[perf["Contest"].eq(pick)].copy()
        by_year = (
            tt.groupby("event_year", observed=True)["net_seconds"]
            .median()
            .reset_index(name="median_net_seconds")
            .sort_values("event_year")
        )
        years_t = sorted([int(y) for y in by_year["event_year"].dropna().unique().tolist()])
        by_year["median_hhmmss"] = by_year["median_net_seconds"].apply(seconds_to_hhmmss)

        fig_time_trend = px.line(by_year, x="event_year", y="median_net_seconds", markers=True, template=PLOTLY_TEMPLATE)
        fig_time_trend.update_traces(
            customdata=np.stack([by_year["median_hhmmss"]], axis=-1),
            hovertemplate=f"<b>{pick}</b><br>Year: %{{x}}<br>Typical finish time: %{{customdata[0]}}<extra></extra>",
        )
        fig_time_trend.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Year",
            yaxis_title="Median finish time (seconds)",
        )
        fix_year_axis(fig_time_trend, years_t)
        apply_modern_hover(fig_time_trend)
        st.plotly_chart(fig_time_trend, width="stretch")

# =========================
# PARA TAB
# =========================
with tab_para:
    st.subheader("Para-athletes (introduced in 2024)")

    para_years = sorted([int(y) for y in para_fin["event_year"].dropna().unique().tolist()])
    if not para_years:
        st.info("No para-athlete rows found in the dataset.")
    else:
        para_fin_2425 = para_fin[para_fin["event_year"].ge(2024)].copy()

        if 2024 in para_years and 2025 in para_years:
            s24 = year_snapshot(para_fin_2425, 2024)
            s25 = year_snapshot(para_fin_2425, 2025)

            d_part = s25["participants"] - s24["participants"]
            growth = (d_part / s24["participants"] * 100) if s24["participants"] else np.nan
            d_pf = (s25["pct_f"] - s24["pct_f"]) * 100 if (not np.isnan(s25["pct_f"]) and not np.isnan(s24["pct_f"])) else np.nan
            d_pm = (s25["pct_m"] - s24["pct_m"]) * 100 if (not np.isnan(s25["pct_m"]) and not np.isnan(s24["pct_m"])) else np.nan
            d_med = s25["med_net"] - s24["med_net"] if (not np.isnan(s25["med_net"]) and not np.isnan(s24["med_net"])) else np.nan

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card(
                    "Participants (2025)",
                    fmt_int(s25["participants"]),
                    sub=f"Δ {fmt_int(d_part)} vs 2024",
                    sub_muted=(f"+{growth:.0f}% growth" if not np.isnan(growth) else ""),
                )
            with c2:
                val = f"{(s25['pct_f']*100):.1f}%" if not np.isnan(s25["pct_f"]) else "—"
                kpi_card("% Female (2025)", val, sub=f"{fmt_pp(d_pf)} vs 2024")
            with c3:
                val = f"{(s25['pct_m']*100):.1f}%" if not np.isnan(s25["pct_m"]) else "—"
                kpi_card("% Male (2025)", val, sub=f"{fmt_pp(d_pm)} vs 2024")
            with c4:
                kpi_card(
                    "Typical finish time (2025)",
                    seconds_to_hhmmss(s25["med_net"]),
                    sub=(f"Δ {seconds_to_hhmmss(d_med)} vs 2024" if not np.isnan(d_med) else ""),
                    sub_muted="(median net time, 2.5K)",
                )

            st.success(
                f"Para participation increased from **{s24['participants']}** (2024) to **{s25['participants']}** (2025). "
                f"That’s **{d_part:+d}** participants (**{growth:.0f}%** growth)."
            )
        else:
            st.info("Para KPIs are designed for 2024 → 2025 (since the division started in 2024).")

        st.divider()

        st.subheader("Participation by disability (2024–2025)")
        d = (
            para_fin_2425.groupby(["disability", "gender"], observed=True)
            .size()
            .reset_index(name="Participants")
            .sort_values("Participants", ascending=False)
        )
        fig_d = px.bar(
            d,
            x="Participants",
            y="disability",
            color="gender",
            orientation="h",
            color_discrete_map=GENDER_COLORS,
            template=PLOTLY_TEMPLATE,
        )
        fig_d.update_traces(
            hovertemplate="<b>%{y}</b><br>Gender: %{fullData.name}<br>Participants: %{x:,}<extra></extra>"
        )
        fig_d.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Participants",
            yaxis_title="Disability",
            legend_title_text="Gender",
        )
        apply_modern_hover(fig_d)
        st.plotly_chart(fig_d, width="stretch")

        st.divider()

        st.subheader("Para participation over time")
        pt = (
            para_fin_2425.groupby("event_year", observed=True)
            .size()
            .reset_index(name="Participants")
            .sort_values("event_year")
        )
        years_p = sorted([int(y) for y in pt["event_year"].dropna().unique().tolist()])
        fig_pt = px.line(pt, x="event_year", y="Participants", markers=True, template=PLOTLY_TEMPLATE)
        fig_pt.update_traces(
            hovertemplate="<b>Para-athletes</b><br>Year: %{x}<br>Participants: %{y:,}<extra></extra>"
        )
        fig_pt.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Year",
            yaxis_title="Participants",
        )
        fix_year_axis(fig_pt, years_p)
        apply_modern_hover(fig_pt)
        st.plotly_chart(fig_pt, width="stretch")