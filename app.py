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
/* Base */
.stApp { background: #f6f9ff; }
h1, h2, h3, h4, h5, h6, p, li, div { color: #0f172a; }

/* Smooth overall scale */
html, body { font-size: 18px; }

/* Section titles */
h2 { font-size: 2.2rem !important; }
h3 { font-size: 1.6rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-size: 1.05rem !important; padding: 10px 14px !important; }

/* Captions + secondary text */
.stCaption, [data-testid="stMarkdownContainer"] p { font-size: 1.02rem; }

/* KPI cards */
.kpi-card{
  background: #ffffff;
  border: 1px solid #e6eefc;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
}
.kpi-label{ font-size: 0.90rem; color: #334155; }
.kpi-value{ font-size: 2.10rem; font-weight: 780; margin-top: 2px; }
.kpi-sub{ font-size: 0.98rem; color: #2563eb; margin-top: 10px; }
.kpi-sub-muted{ font-size: 0.90rem; color: #64748b; margin-top: 4px; }

/* Softer plotly modebar */
.modebar{ opacity: 0.18 !important; }
.modebar:hover{ opacity: 1 !important; }

/* HERO (single cyan) */
.hero {
  border-radius: 18px;
  padding: 18px 18px;
  color: white;
  background: #06b6d4; /* cyan */
  border: 1px solid rgba(255,255,255,0.22);
  box-shadow: 0 10px 30px rgba(2, 132, 199, 0.18);
}
.hero-title { font-size: 1.05rem; opacity: 0.95; font-weight: 750; }
.hero-big { font-size: 2.6rem; font-weight: 900; margin-top: 8px; line-height: 1.05; }
.hero-sub { margin-top: 10px; font-size: 1.05rem; opacity: 0.95; }

/* Hero chips */
.hero-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 14px; }
.hero-chip {
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 14px;
  padding: 12px 14px;
}
.hero-chip-label { font-size: 0.95rem; opacity: 0.95; }
.hero-chip-value { font-size: 1.35rem; font-weight: 900; margin-top: 6px; }

/* Context blocks */
div[data-testid="stAlert"]{
  border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"

# Blue + Green palette (no red)
GENDER_COLORS = {"Female": "#2563eb", "Male": "#22c55e"}

# =========================
# Plot styling helpers
# =========================
def apply_modern_hover(fig):
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#e2e8f0",
            font=dict(size=14, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"),
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


def style_fig(fig, title=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(size=16, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"),
        legend=dict(font=dict(size=14)),
        title=dict(text=title or "", font=dict(size=22)),
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
    )
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    return fig


# =========================
# Paths + data loading
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "nbk_run_2021_to_2025.csv"


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

    # contest normalize
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


def dedupe_finishers(frame: pd.DataFrame) -> pd.DataFrame:
    """Finishers = unique (year, contest, division, bib)."""
    x = frame.copy()
    if x["bib"].notna().any():
        x = x.dropna(subset=["bib"])
        x = x.drop_duplicates(subset=["event_year", "contest", "division", "bib"], keep="first")
    return x


def year_snapshot(fin: pd.DataFrame, year: int) -> dict:
    y = fin[fin["event_year"].eq(year)].copy()
    participants = len(y)
    if participants == 0:
        return dict(participants=0, pct_f=np.nan, pct_m=np.nan)

    pct_f = y["gender"].eq("Female").mean()
    pct_m = y["gender"].eq("Male").mean()
    return dict(participants=participants, pct_f=pct_f, pct_m=pct_m)


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
    """
    11K isn't a standard race distance, so we group it with 10K in most views.
    Exception: when viewing 2021 alone in 'At a glance', we can show 11K explicitly.
    """
    s = series.astype("string").fillna("").astype(str)
    if keep_11k:
        return s
    return s.replace({"11K": "10K"})


def fmt_change_count(prev: int, curr: int, prev_year: int) -> str:
    """Human-friendly change text for counts."""
    if prev is None or prev == 0:
        return f"In {prev_year} there was no baseline to compare."
    delta = curr - prev
    pct = (delta / prev) * 100 if prev else np.nan
    if delta > 0:
        return f"Up by {fmt_int(delta)} (+{pct:.1f}%) vs {prev_year}"
    if delta < 0:
        return f"Down by {fmt_int(abs(delta))} ({pct:.1f}%) vs {prev_year}"
    return f"No change vs {prev_year}"


def fmt_change_share(prev_share: float, curr_share: float, prev_year: int) -> str:
    """Human-friendly change text for shares (percentage points)."""
    if prev_share is None or curr_share is None or np.isnan(prev_share) or np.isnan(curr_share):
        return f"Not enough data to compare vs {prev_year}"
    prev_pct = prev_share * 100
    curr_pct = curr_share * 100
    pp = curr_pct - prev_pct
    direction = "up" if pp > 0 else "down" if pp < 0 else "unchanged"
    pp_abs = abs(pp)
    return f"{prev_pct:.1f}% → {curr_pct:.1f}% ({direction} {pp_abs:.1f} percentage points) vs {prev_year}"


# =========================
# Load
# =========================
if not DEFAULT_CSV.exists():
    st.error(
        "Could not load data.\n\n"
        f"Expected file at:\n{DEFAULT_CSV}\n\n"
        "Fix: ensure your CSV is committed at `data/nbk_run_2021_to_2025.csv`."
    )
    st.stop()

df = prepare(load_csv(DEFAULT_CSV))

years_all = sorted([int(y) for y in df["event_year"].dropna().unique().tolist()])
y_latest = max(years_all) if years_all else None
y_prev = years_all[-2] if len(years_all) >= 2 else None

# =========================
# Header
# =========================
st.title("NBK Run Dashboard")
st.caption("For the last four annual runs.")

# =========================
# Split: Open / Para
# =========================
OPEN_CONTESTS_RAW = ["5K", "10K", "11K", "21K"]
OPEN_CONTESTS_BUCKETED = ["5K", "10K", "21K"]  # 11K grouped into 10K in most views

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

    # --- Story hero (clear language) ---
    if y_latest is not None and y_prev is not None:
        fin_by_year = (
            open_fin.groupby("event_year", observed=True)
            .size()
            .reset_index(name="Finishers")
            .sort_values("event_year")
        )

        latest_finishers = int(fin_by_year.loc[fin_by_year["event_year"].eq(y_latest), "Finishers"].iloc[0]) if fin_by_year["event_year"].eq(y_latest).any() else 0
        prev_finishers = int(fin_by_year.loc[fin_by_year["event_year"].eq(y_prev), "Finishers"].iloc[0]) if fin_by_year["event_year"].eq(y_prev).any() else 0

        is_record = (latest_finishers == fin_by_year["Finishers"].max()) if not fin_by_year.empty else False
        # record_text = "🏆 Record year — highest Open finishers across the last four runs." if is_record else "Open finishers snapshot."
        record_text = "🏆 Record year — highest finishers across the last four runs."


        hero_chip_1_label = f"{y_prev} → {y_latest}"
        hero_chip_1_value = f"{fmt_int(prev_finishers)} → {fmt_int(latest_finishers)}"

        hero_chip_2_label = f"Compared with {y_prev}"
        hero_chip_2_value = fmt_change_count(prev_finishers, latest_finishers, y_prev)

        st.markdown(
            f"""
<div class="hero">
  <div class="hero-title">Open division — {y_latest} snapshot</div>
  <div class="hero-big">{fmt_int(latest_finishers)} finishers</div>
  <div class="hero-sub">{record_text}</div>

  <div class="hero-grid">
    <div class="hero-chip">
      <div class="hero-chip-label">{hero_chip_1_label}</div>
      <div class="hero-chip-value">{hero_chip_1_value}</div>
    </div>
    <div class="hero-chip">
      <div class="hero-chip-label">{hero_chip_2_label}</div>
      <div class="hero-chip-value">{hero_chip_2_value}</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Not enough yearly data to compute year-over-year insights.")

    st.divider()

    # --- KPI cards (make change very obvious, not shorthand) ---
    if y_latest is None or y_prev is None:
        st.info("Not enough yearly data to compute year-over-year KPIs.")
    else:
        s_latest = year_snapshot(open_fin, y_latest)
        s_prev = year_snapshot(open_fin, y_prev)

        # c1, c2, c3 = st.columns(3)
        c2, c3 = st.columns(2)


        # with c1:
        #     kpi_card(
        #         f"Finishers ({y_latest})",
        #         fmt_int(s_latest["participants"]),
        #         sub=fmt_change_count(int(s_prev["participants"]), int(s_latest["participants"]), y_prev),
        #     )

        with c2:
            val = f"{(s_latest['pct_f']*100):.1f}%" if not np.isnan(s_latest["pct_f"]) else "—"
            kpi_card(
                f"Female share ({y_latest})",
                val,
                sub=fmt_change_share(float(s_prev["pct_f"]), float(s_latest["pct_f"]), y_prev),
            )

        with c3:
            val = f"{(s_latest['pct_m']*100):.1f}%" if not np.isnan(s_latest["pct_m"]) else "—"
            kpi_card(
                f"Male share ({y_latest})",
                val,
                sub=fmt_change_share(float(s_prev["pct_m"]), float(s_latest["pct_m"]), y_prev),
            )

    st.divider()

    # =========================
    # At a glance (distribution)
    # =========================
    st.subheader("At a glance")

    year_opts = [y_latest] + [y for y in years_all if y != y_latest]
    at_year = st.selectbox(
        "Show distributions for",
        options=(["All years"] + year_opts),
        index=1,
        help="Defaults to the most recent year so the distribution reflects the latest run.",
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
        .reset_index(name="Finishers")
        .query("Finishers > 0")
    )

    fig_donut = px.pie(share, names="Contest", values="Finishers", hole=0.55)
    fig_donut.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Finishers: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    style_fig(fig_donut)
    apply_modern_hover(fig_donut)

    gbar = (
        af.groupby(["Contest", "gender"], observed=True)
        .size()
        .reset_index(name="Finishers")
    )

    fig_bar = px.bar(
        gbar,
        x="Contest",
        y="Finishers",
        color="gender",
        barmode="group",
        color_discrete_map=GENDER_COLORS,
        category_orders={"Contest": contest_order_here},
    )
    fig_bar.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Contest: %{x}<br>Finishers: %{y:,}<extra></extra>"
    )
    fig_bar.update_layout(xaxis_title="Contest", yaxis_title="Finishers")
    style_fig(fig_bar)
    apply_modern_hover(fig_bar)

    with left:
        st.plotly_chart(fig_donut, width="stretch")
        st.caption(f"Contest mix for **{at_year_label}** (Open division).")

    with right:
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(f"Finishers by contest in **{at_year_label}**, split by gender.")

    # Context messages ONLY when "All years" selected (per your request)
    if at_year == "All years":
        st.info("Context: **11K** (seen in 2021) is grouped into **10K** in this dashboard for clearer analysis.")
        st.info("Context: **21K** was introduced in **2025** (so it won’t appear in earlier-year distributions).")

    st.divider()

    # =========================
    # Contest share over time (100% stacked)
    # =========================
    st.subheader("Contest share over time (Open)")

    tmp = open_fin.assign(contest_bucket=bucket_open_contest(open_fin["contest"], keep_11k=False))
    counts = (
        tmp.groupby(["event_year", "contest_bucket"], observed=True)
        .size()
        .reset_index(name="Finishers")
    )
    totals = counts.groupby("event_year")["Finishers"].sum().reset_index(name="Total")
    share_over_time = counts.merge(totals, on="event_year", how="left")
    share_over_time["Share"] = np.where(
        share_over_time["Total"] > 0,
        share_over_time["Finishers"] / share_over_time["Total"],
        np.nan,
    )

    fig_share_contest = px.bar(
        share_over_time,
        x="event_year",
        y="Share",
        color="contest_bucket",
        barmode="stack",
        category_orders={"contest_bucket": OPEN_CONTESTS_BUCKETED},
    )
    fig_share_contest.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1%}<extra></extra>"
    )
    fig_share_contest.update_layout(
        xaxis_title="Year",
        yaxis_title="Share of Open finishers",
        legend_title_text="Contest (11K → 10K)",
    )
    fix_year_axis(fig_share_contest, sorted(share_over_time["event_year"].dropna().unique().tolist()))
    style_fig(fig_share_contest)
    apply_modern_hover(fig_share_contest)
    st.plotly_chart(fig_share_contest, width="stretch")

    st.info("Context: This chart shows **percentage share** (not counts). It highlights how the contest mix shifts year-to-year.")

    st.divider()

    # =========================
    # YoY change by contest (latest vs previous)
    # =========================
    st.subheader(f"Year-over-year change by contest ({y_latest} vs {y_prev})")

    if y_latest is None or y_prev is None:
        st.info("Not enough yearly data to compute YoY changes.")
    else:
        a = tmp[tmp["event_year"].eq(y_latest)].copy()
        b = tmp[tmp["event_year"].eq(y_prev)].copy()

        a_counts = a.groupby("contest_bucket", observed=True).size().reset_index(name=str(y_latest))
        b_counts = b.groupby("contest_bucket", observed=True).size().reset_index(name=str(y_prev))

        yoy = a_counts.merge(b_counts, on="contest_bucket", how="outer").fillna(0)
        yoy["Change"] = yoy[str(y_latest)] - yoy[str(y_prev)]

        # continuing contests = present in both years
        continuing = yoy[(yoy[str(y_latest)] > 0) & (yoy[str(y_prev)] > 0)].copy()
        all_continuing_increased = bool((continuing["Change"] > 0).all()) if not continuing.empty else False

        # new contests (e.g., 21K in 2025)
        new_contests = yoy[(yoy[str(y_latest)] > 0) & (yoy[str(y_prev)] == 0)].copy()

        yoy = yoy.sort_values("contest_bucket")
        yoy["Change_label"] = yoy["Change"].apply(lambda v: f"{v:+.0f}")

        # Bar chart of YoY change
        fig_yoy = px.bar(
            yoy,
            x="contest_bucket",
            y="Change",
            text="Change_label",
            category_orders={"contest_bucket": OPEN_CONTESTS_BUCKETED},
        )
        fig_yoy.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>"
                          f"{y_prev}: %{{customdata[0]:,.0f}}<br>"
                          f"{y_latest}: %{{customdata[1]:,.0f}}<br>"
                          "Change: %{y:+,.0f}<extra></extra>",
            customdata=np.stack([yoy[str(y_prev)].values, yoy[str(y_latest)].values], axis=-1),
        )
        fig_yoy.update_layout(
            xaxis_title="",
            yaxis_title=f"Change in finishers ({y_latest} minus {y_prev})",
        )
        style_fig(fig_yoy)
        apply_modern_hover(fig_yoy)
        st.plotly_chart(fig_yoy, width="stretch")

        # Make the message explicit (your request)
        if all_continuing_increased:
            st.success(
                f"**All contests that existed in {y_prev} had more Open finishers in {y_latest}.** "
                f"Also, **21K was introduced in {y_latest}**, so it has no {y_prev} baseline."
            )
        else:
            st.info(
                f"These bars show how each contest changed from {y_prev} to {y_latest}. "
                f"Note: **21K was introduced in {y_latest}**."
            )

        # Show a clean table for clarity (people love this on LinkedIn)
        view = yoy.rename(columns={"contest_bucket": "Contest"})
        view["Contest"] = view["Contest"].astype(str)
        view = view[["Contest", str(y_prev), str(y_latest), "Change"]]
        st.dataframe(view, width="stretch", hide_index=True)

        if not new_contests.empty:
            new_list = ", ".join(new_contests["contest_bucket"].astype(str).tolist())
            st.info(f"New in {y_latest}: **{new_list}**.")

    st.divider()

    # =========================
    # Participation over time (counts)
    # =========================
    st.subheader("Participation over time (Open)")

    trend = (
        tmp.groupby(["event_year", "contest_bucket"], observed=True)
        .size()
        .reset_index(name="Finishers")
        .sort_values(["event_year", "contest_bucket"])
    )

    fig_trend = px.line(
        trend,
        x="event_year",
        y="Finishers",
        color="contest_bucket",
        markers=True,
        category_orders={"contest_bucket": OPEN_CONTESTS_BUCKETED},
    )
    fig_trend.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Finishers: %{y:,}<extra></extra>"
    )
    fig_trend.update_layout(xaxis_title="Year", yaxis_title="Finishers", legend_title_text="")
    fix_year_axis(fig_trend, sorted(trend["event_year"].dropna().unique().tolist()))
    style_fig(fig_trend)
    apply_modern_hover(fig_trend)
    st.plotly_chart(fig_trend, width="stretch")

    st.info("Context: **11K** appears only in **2021** and is grouped into **10K** here. **21K** appears only in **2025**.")

    st.divider()

    # =========================
    # Gender balance (counts + shares)
    # =========================
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

    # stacked counts with % in hover
    fig_counts = go.Figure()
    fig_counts.add_trace(
        go.Bar(
            x=pivot["event_year"],
            y=pivot["Female"],
            name="Female",
            marker_color=GENDER_COLORS["Female"],
            customdata=np.stack([pivot["Female_pct"] * 100], axis=-1),
            hovertemplate="<b>Female</b><br>Year: %{x}<br>Finishers: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig_counts.add_trace(
        go.Bar(
            x=pivot["event_year"],
            y=pivot["Male"],
            name="Male",
            marker_color=GENDER_COLORS["Male"],
            customdata=np.stack([pivot["Male_pct"] * 100], axis=-1),
            hovertemplate="<b>Male</b><br>Year: %{x}<br>Finishers: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig_counts.update_layout(barmode="stack", xaxis_title="Year", yaxis_title="Finishers", legend_title_text="Gender")
    fix_year_axis(fig_counts, years_g)
    style_fig(fig_counts)
    apply_modern_hover(fig_counts)
    st.plotly_chart(fig_counts, width="stretch")
    st.caption("Hover any bar segment to see both **finishers** and **share (%)**.")

    # share lines: Female + Male
    fig_share = go.Figure()
    fig_share.add_trace(
        go.Scatter(
            x=pivot["event_year"],
            y=pivot["Female_pct"] * 100,
            mode="lines+markers",
            name="Female share",
            hovertemplate="<b>Female share</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>",
            line=dict(color=GENDER_COLORS["Female"]),
        )
    )
    fig_share.add_trace(
        go.Scatter(
            x=pivot["event_year"],
            y=pivot["Male_pct"] * 100,
            mode="lines+markers",
            name="Male share",
            hovertemplate="<b>Male share</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>",
            line=dict(color=GENDER_COLORS["Male"]),
        )
    )
    fig_share.update_layout(xaxis_title="Year", yaxis_title="Share (%)", legend_title_text="")
    fix_year_axis(fig_share, years_g)
    style_fig(fig_share)
    apply_modern_hover(fig_share)
    st.plotly_chart(fig_share, width="stretch")

    st.divider()

    # =========================
    # Gender share by contest (latest vs previous)
    # =========================
    st.subheader(f"Female share by contest ({y_latest} vs {y_prev})")

    if y_latest is None or y_prev is None:
        st.info("Not enough yearly data to compute this view.")
    else:
        def contest_gender_share(frame, year):
            x = frame[frame["event_year"].eq(year)].copy()
            x = x.assign(contest_bucket=bucket_open_contest(x["contest"], keep_11k=False))
            c = (
                x.groupby(["contest_bucket", "gender"], observed=True)
                .size()
                .reset_index(name="count")
            )
            t = c.groupby("contest_bucket")["count"].sum().reset_index(name="total")
            c = c.merge(t, on="contest_bucket", how="left")
            c["share"] = np.where(c["total"] > 0, c["count"] / c["total"], np.nan)
            c["year"] = str(year)
            return c

        s_a = contest_gender_share(open_fin, y_latest)
        s_b = contest_gender_share(open_fin, y_prev)
        s = pd.concat([s_a, s_b], ignore_index=True)
        s_f = s[s["gender"].eq("Female")].copy()

        year_colors = {str(y_prev): "#94a3b8", str(y_latest): "#2563eb"}  # gray vs blue
        fig_gcs = px.bar(
            s_f,
            x="contest_bucket",
            y="share",
            color="year",
            barmode="group",
            category_orders={"contest_bucket": OPEN_CONTESTS_BUCKETED},
            color_discrete_map=year_colors,
        )
        fig_gcs.update_traces(
            hovertemplate="<b>Female share</b><br>Contest: %{x}<br>Share: %{y:.1%}<extra></extra>"
        )
        fig_gcs.update_layout(xaxis_title="", yaxis_title="Female share", legend_title_text="Year")
        style_fig(fig_gcs)
        apply_modern_hover(fig_gcs)
        st.plotly_chart(fig_gcs, width="stretch")

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

            st.markdown(
                f"""
<div class="hero">
  <div class="hero-title">Para-athletes — 2024 → 2025 growth</div>
  <div class="hero-big">{fmt_int(s25["participants"])} participants (2025)</div>
  <div class="hero-sub">{fmt_change_count(int(s24["participants"]), int(s25["participants"]), 2024)}</div>
  <div class="hero-grid">
    <div class="hero-chip">
      <div class="hero-chip-label">2024 → 2025</div>
      <div class="hero-chip-value">{fmt_int(s24["participants"])} → {fmt_int(s25["participants"])}</div>
    </div>
    <div class="hero-chip">
      <div class="hero-chip-label">Division launched</div>
      <div class="hero-chip-value">2024</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

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
        )
        fig_d.update_traces(
            hovertemplate="<b>%{y}</b><br>Gender: %{fullData.name}<br>Participants: %{x:,}<extra></extra>"
        )
        fig_d.update_layout(xaxis_title="Participants", yaxis_title="Disability", legend_title_text="Gender")
        style_fig(fig_d)
        apply_modern_hover(fig_d)
        st.plotly_chart(fig_d, width="stretch")

        st.divider()

        st.subheader("Disability growth (2025 vs 2024)")

        if 2024 in para_years and 2025 in para_years:
            c24 = para_fin_2425[para_fin_2425["event_year"].eq(2024)].groupby("disability").size().reset_index(name="2024")
            c25 = para_fin_2425[para_fin_2425["event_year"].eq(2025)].groupby("disability").size().reset_index(name="2025")
            dg = c25.merge(c24, on="disability", how="outer").fillna(0)
            dg["Change"] = dg["2025"] - dg["2024"]
            dg = dg.sort_values("Change", ascending=False)

            dg["label"] = dg["Change"].apply(lambda v: f"{v:+.0f}")

            fig_dg = px.bar(
                dg,
                x="Change",
                y="disability",
                orientation="h",
                text="label",
            )
            fig_dg.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>2024: %{customdata[0]:.0f}<br>2025: %{customdata[1]:.0f}<br>Change: %{x:+.0f}<extra></extra>",
                customdata=np.stack([dg["2024"].values, dg["2025"].values], axis=-1),
            )
            fig_dg.update_layout(xaxis_title="Change in participants (2025 minus 2024)", yaxis_title="Disability")
            style_fig(fig_dg)
            apply_modern_hover(fig_dg)
            st.plotly_chart(fig_dg, width="stretch")
        else:
            st.info("This chart requires both 2024 and 2025 para data.")