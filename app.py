"""
LEDGER — Smart Personal Finance Tracker
========================================
A black-and-gold Streamlit dashboard with:

  Dashboard      — KPI strip, donut, monthly trend, recent transactions
  Analysis       — Category deep-dive, treemap, month-over-month heatmap
  Cash Flow      — Waterfall chart, income vs expense stream, daily burn
  Predictions    — Linear-regression 3-month forecast + confidence band
  Insights       — Rule-engine personalised suggestions + 50/30/20 check
  Transactions   — Filterable table, inline delete, CSV export

Setup
-----
    pip install streamlit pandas numpy plotly scikit-learn
Run
---
    streamlit run app.py
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json, os
from datetime import date, timedelta

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be the very first Streamlit call
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ledger",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ════════════════════════════════════════════════════════════════════════════
GOLD        = "#C9A84C"   # antique gold — the single accent
GOLD_DIM    = "#8A6E2F"   # muted gold for secondary uses
GOLD_LIGHT  = "#E8D5A3"   # light gold for hover / highlight text
BLACK       = "#0A0A0A"   # void black background
CARD        = "#111111"   # card / panel background
BORDER      = "#1E1E1E"   # hairline dividers
MUTED       = "#6B6B6B"   # secondary text
WHITE       = "#F5F5F0"   # warm white primary text
GREEN       = "#4CAF7D"   # positive delta
RED         = "#CF6679"   # negative delta / over-budget

# Category colour map — tonal, no neon
CAT_COLOURS = {
    "Food & Dining":    "#C9A84C",
    "Transport":        "#8A9BB5",
    "Shopping":         "#A89BC0",
    "Entertainment":    "#7EB5A6",
    "Health":           "#7DAF7D",
    "Utilities":        "#B5A97E",
    "Rent & Housing":   "#C09080",
    "Education":        "#8AABB5",
    "Personal Care":    "#B58A9B",
    "Investment":       "#9BB58A",
    "Salary":           "#4CAF7D",
    "Freelance":        "#6B9FBF",
    "Other Income":     "#9FAFBF",
    "Miscellaneous":    "#6B6B6B",
}

# ════════════════════════════════════════════════════════════════════════════
#  CATEGORIES & KEYWORDS (for auto-categorisation)
# ════════════════════════════════════════════════════════════════════════════
CATEGORIES = {
    "Food & Dining":    ["zomato","swiggy","restaurant","food","cafe","coffee",
                         "lunch","dinner","breakfast","grocery","supermarket",
                         "blinkit","zepto","dunzo","hotel","pizza","burger"],
    "Transport":        ["uber","ola","petrol","fuel","metro","bus","auto",
                         "cab","parking","rapido","irctc","train","flight",
                         "rapido","namma","bmtc","redbus"],
    "Shopping":         ["amazon","flipkart","myntra","ajio","mall","shop",
                         "meesho","nykaa","purchase","buy","store","market"],
    "Entertainment":    ["netflix","prime","spotify","hotstar","disney",
                         "movie","game","youtube","bookmyshow","theater",
                         "concert","event"],
    "Health":           ["pharmacy","doctor","hospital","clinic","medicine",
                         "apollo","medplus","healthkart","lab","test","scan"],
    "Utilities":        ["electricity","water","gas","internet","wifi",
                         "broadband","recharge","dth","jio","airtel","vi",
                         "bsnl","tata sky"],
    "Rent & Housing":   ["rent","maintenance","housing","apartment","society",
                         "landlord","deposit","lease"],
    "Education":        ["course","udemy","coursera","book","tuition","school",
                         "college","fee","udacity","skillshare","pluralsight"],
    "Personal Care":    ["salon","haircut","spa","gym","fitness","grooming",
                         "parlour","laundry"],
    "Investment":       ["mutual fund","sip","stocks","zerodha","groww",
                         "crypto","fd","nps","ppf","smallcase","etf"],
    "Salary":           ["salary","payroll","ctc","stipend","wages","hike"],
    "Freelance":        ["freelance","project","client","invoice",
                         "payment received","transfer received","consulting"],
    "Other Income":     ["cashback","refund","gift","bonus","interest",
                         "dividend","award","prize"],
    "Miscellaneous":    [],
}

EXPENSE_CATS = [c for c in CATEGORIES
                if c not in ("Salary","Freelance","Other Income")]
INCOME_CATS  = ["Salary","Freelance","Other Income"]
DATA_FILE    = "transactions.json"

# ════════════════════════════════════════════════════════════════════════════
#  CSS — Ledger theme
#  Cormorant Garamond: display / large numbers only
#  Inter: everything else
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ─────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {WHITE};
}}
.stApp {{
    background-color: {BLACK};
}}
.block-container {{
    padding: 0 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label span {{
    color: {WHITE} !important;
    font-size: 0.82rem;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent;
    border: 1px solid {GOLD};
    color: {GOLD} !important;
    border-radius: 2px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.55rem 1rem;
    width: 100%;
    transition: background 0.2s, color 0.2s;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {GOLD};
    color: {BLACK} !important;
}}

/* ── Main button ──────────────────────────────────────────── */
.main .stButton > button {{
    background: transparent;
    border: 1px solid {GOLD};
    color: {GOLD} !important;
    border-radius: 2px;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.45rem 1.2rem;
    transition: background 0.2s, color 0.2s;
}}
.main .stButton > button:hover {{
    background: {GOLD};
    color: {BLACK} !important;
}}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, select,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: {BLACK} !important;
    border-color: {BORDER} !important;
    color: {WHITE} !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
}}
input:focus, div[data-baseweb="select"] > div:focus-within {{
    border-color: {GOLD} !important;
    box-shadow: 0 0 0 1px {GOLD} !important;
}}
label {{ color: {MUTED} !important; font-size: 0.75rem !important;
         letter-spacing: 0.05em; text-transform: uppercase; }}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    border-radius: 0;
    border-bottom: 2px solid transparent;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.7rem 1.4rem;
    margin-bottom: -1px;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
    background: transparent !important;
}}

/* ── KPI card ─────────────────────────────────────────────── */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 2px solid {GOLD};
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 0;
}}
.kpi-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.5rem;
}}
.kpi-value {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 400;
    color: {WHITE};
    line-height: 1;
    letter-spacing: -0.01em;
}}
.kpi-delta {{
    font-size: 0.73rem;
    font-weight: 500;
    margin-top: 0.4rem;
    letter-spacing: 0.02em;
}}
.kpi-up   {{ color: {GREEN}; }}
.kpi-down {{ color: {RED};   }}
.kpi-flat {{ color: {MUTED}; }}

/* ── Section rule ─────────────────────────────────────────── */
.ledger-rule {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.8rem 0 1.1rem;
}}
.ledger-rule-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {GOLD};
    white-space: nowrap;
}}
.ledger-rule-line {{
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

/* ── Insight card ─────────────────────────────────────────── */
.ins-card {{
    border-left: 2px solid;
    background: {CARD};
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    font-size: 0.85rem;
    color: {WHITE};
    line-height: 1.6;
    border-radius: 0 2px 2px 0;
}}
.ins-success {{ border-color: {GREEN};        }}
.ins-warning {{ border-color: {GOLD};         }}
.ins-danger  {{ border-color: {RED};          }}
.ins-info    {{ border-color: #8A9BB5;        }}
.ins-title   {{ font-weight: 600; color: {GOLD_LIGHT}; margin-bottom: 0.2rem; font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; }}

/* ── Tip card ─────────────────────────────────────────────── */
.tip-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: {WHITE};
    line-height: 1.6;
}}
.tip-title {{
    font-weight: 600;
    color: {GOLD};
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}}

/* ── Dataframe ────────────────────────────────────────────── */
.stDataFrame {{
    border: 1px solid {BORDER};
    border-radius: 2px;
}}
.stDataFrame th {{
    background: {CARD} !important;
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid {BORDER} !important;
}}
.stDataFrame td {{
    font-size: 0.83rem !important;
    color: {WHITE} !important;
    background: {BLACK} !important;
    border-bottom: 1px solid {BORDER} !important;
}}

/* ── Page header ──────────────────────────────────────────── */
.page-header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    padding: 2rem 0 0.5rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0.5rem;
}}
.page-wordmark {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {GOLD};
}}
.page-sub {{
    font-size: 0.73rem;
    color: {MUTED};
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* ── Selectbox dropdown ───────────────────────────────────── */
div[data-baseweb="popover"] ul {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
}}
div[data-baseweb="popover"] li {{
    color: {WHITE} !important;
    font-size: 0.84rem !important;
}}
div[data-baseweb="popover"] li:hover {{
    background: {BORDER} !important;
}}

/* ── Radio ────────────────────────────────────────────────── */
.stRadio > div {{ gap: 1rem; }}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BLACK}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {GOLD_DIM}; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load transactions from JSON file; return empty frame if none exist."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            records = json.load(f)
        if records:
            df = pd.DataFrame(records)
            df["date"]   = pd.to_datetime(df["date"])
            df["amount"] = df["amount"].astype(float)
            return df.sort_values("date", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["id","date","type","amount","category","note"])


def save_data(df: pd.DataFrame):
    """Persist DataFrame to JSON, converting Timestamps to ISO strings."""
    out = df.copy()
    out["date"] = out["date"].astype(str)
    with open(DATA_FILE, "w") as f:
        json.dump(out.to_dict("records"), f, indent=2, default=str)


def next_id(df: pd.DataFrame) -> int:
    return int(df["id"].max()) + 1 if not df.empty else 1


# ════════════════════════════════════════════════════════════════════════════
#  AUTO-CATEGORISATION
#  Keyword scan of description text → best category match.
# ════════════════════════════════════════════════════════════════════════════

def auto_categorise(note: str, t_type: str) -> str:
    text = (note or "").lower()
    # Score each category by keyword hits
    best_cat, best_score = "Miscellaneous", 0
    cat_pool = INCOME_CATS if t_type == "income" else EXPENSE_CATS
    for cat in cat_pool:
        score = sum(1 for kw in CATEGORIES[cat] if kw in text)
        if score > best_score:
            best_score, best_cat = score, cat
    if best_score == 0:
        best_cat = "Salary" if t_type == "income" else "Miscellaneous"
    return best_cat


# ════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per calendar month:
    columns — month (str), income, expense, net, savings_rate
    """
    if df.empty:
        return pd.DataFrame(columns=["month","income","expense","net","savings_rate"])
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M")
    g = (tmp.groupby(["month","type"])["amount"]
           .sum().unstack(fill_value=0))
    g.columns.name = None
    for col in ("income","expense"):
        if col not in g.columns:
            g[col] = 0.0
    g = g[["income","expense"]].reset_index()
    g["month"] = g["month"].astype(str)
    g["net"]   = g["income"] - g["expense"]
    g["savings_rate"] = np.where(
        g["income"] > 0,
        (g["net"] / g["income"] * 100).round(1),
        0.0,
    )
    return g.sort_values("month").reset_index(drop=True)


def category_spend(df: pd.DataFrame, month: str = None) -> pd.DataFrame:
    """Expense totals by category, optionally filtered to a YYYY-MM month."""
    tmp = df[df["type"] == "expense"].copy()
    if month:
        tmp = tmp[tmp["date"].dt.to_period("M").astype(str) == month]
    if tmp.empty:
        return pd.DataFrame(columns=["category","amount"])
    return (tmp.groupby("category")["amount"].sum()
              .reset_index()
              .sort_values("amount", ascending=False))


def daily_spend(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Daily expense totals for a given month (YYYY-MM)."""
    tmp = df[(df["type"] == "expense") &
             (df["date"].dt.to_period("M").astype(str) == month)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["date","amount"])
    return tmp.groupby("date")["amount"].sum().reset_index().sort_values("date")


def predict_expenses(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Linear regression on monthly expense totals → forecast n months ahead.
    Returns DataFrame with columns: month, predicted, lower, upper
    (±1 std dev confidence band).
    Needs ≥ 3 months of data.
    """
    ms = monthly_summary(df)
    if len(ms) < 3:
        return pd.DataFrame()
    ms = ms.reset_index(drop=True)
    ms["x"] = np.arange(len(ms))
    X, y = ms[["x"]].values, ms["expense"].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    std_err   = residuals.std()
    last_x    = ms["x"].max()
    last_p    = pd.Period(ms["month"].iloc[-1], "M")
    rows = []
    for i in range(1, n + 1):
        pred = max(model.predict([[last_x + i]])[0], 0)
        rows.append({
            "month":     (last_p + i).strftime("%Y-%m"),
            "predicted": round(pred, 2),
            "lower":     round(max(pred - std_err, 0), 2),
            "upper":     round(pred + std_err, 2),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHT ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> list:
    """
    Rule-based financial insight cards.
    Each dict: {kind, title, body}
    kind ∈ success | warning | danger | info
    """
    if df.empty:
        return [{"kind":"info","title":"Getting started",
                 "body":"Add your first transaction. Insights will appear as you build your history."}]
    ms       = monthly_summary(df)
    now      = pd.Timestamp.now()
    cur_m    = now.to_period("M").strftime("%Y-%m")
    prev_m   = (now - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")
    cur_row  = ms[ms["month"] == cur_m]
    prev_row = ms[ms["month"] == prev_m]

    def _val(row, col): return row[col].values[0] if not row.empty else 0.0

    income   = _val(cur_row,  "income")
    expense  = _val(cur_row,  "expense")
    net      = _val(cur_row,  "net")
    sr       = _val(cur_row,  "savings_rate")
    p_exp    = _val(prev_row, "expense")

    insights = []

    # — Savings rate ─────────────────────────────────────
    if income > 0:
        if sr >= 30:
            insights.append({"kind":"success","title":"Strong savings rate",
                "body":f"You saved {sr:.1f}% of your income this month — well above the 20% benchmark. Maintain the discipline."})
        elif sr >= 20:
            insights.append({"kind":"success","title":"On target",
                "body":f"Savings rate: {sr:.1f}%. You're hitting the 20% goal. Consider routing the surplus into an index fund SIP."})
        elif sr >= 0:
            shortfall = (0.20 * income) - net
            insights.append({"kind":"warning","title":"Below savings target",
                "body":f"Your savings rate is {sr:.1f}%. Cutting ₹{shortfall:,.0f} from discretionary spending would bring you to 20%."})
        else:
            insights.append({"kind":"danger","title":"Spending exceeds income",
                "body":f"Deficit of ₹{abs(net):,.0f} this month. Review your top three expense categories immediately."})

    # — MoM expense change ───────────────────────────────
    if p_exp > 0:
        chg = (expense - p_exp) / p_exp * 100
        if chg > 20:
            insights.append({"kind":"danger","title":"Expenses spiked this month",
                "body":f"Spending is up {chg:.1f}% vs last month (₹{p_exp:,.0f} → ₹{expense:,.0f}). Identify what drove the increase."})
        elif chg < -10:
            insights.append({"kind":"success","title":"Spending reduced",
                "body":f"Expenses fell {abs(chg):.1f}% compared to last month. Good control."})

    # — Dominant category ────────────────────────────────
    cat_df = category_spend(df, month=cur_m)
    if not cat_df.empty and income > 0:
        top_cat = cat_df.iloc[0]["category"]
        top_amt = cat_df.iloc[0]["amount"]
        top_pct = top_amt / income * 100
        if top_pct > 35:
            insights.append({"kind":"warning","title":f"Heavy spend on {top_cat}",
                "body":f"{top_cat} is consuming {top_pct:.1f}% of your income (₹{top_amt:,.0f}). Audit this category."})

    # — Food heuristic ───────────────────────────────────
    food = cat_df[cat_df["category"]=="Food & Dining"]["amount"].sum() if not cat_df.empty else 0
    if income > 0 and food / income > 0.25:
        insights.append({"kind":"warning","title":"Food spend is high",
            "body":f"Food & Dining is {food/income*100:.1f}% of income. Meal prepping 3 days a week typically reduces this by 25–35%."})

    # — Subscription audit ───────────────────────────────
    ent = cat_df[cat_df["category"]=="Entertainment"]["amount"].sum() if not cat_df.empty else 0
    if income > 0 and ent / income > 0.12:
        insights.append({"kind":"info","title":"Review subscriptions",
            "body":f"Entertainment is {ent/income*100:.1f}% of income. Audit active streaming and app subscriptions — unused ones accumulate silently."})

    # — 50/30/20 rule ─────────────────────────────────────
    if income > 0:
        needs_cats = ["Rent & Housing","Utilities","Food & Dining","Health","Transport"]
        wants_cats = ["Entertainment","Shopping","Personal Care"]
        needs = cat_df[cat_df["category"].isin(needs_cats)]["amount"].sum() if not cat_df.empty else 0
        wants = cat_df[cat_df["category"].isin(wants_cats)]["amount"].sum() if not cat_df.empty else 0
        np_   = needs / income * 100
        wp_   = wants / income * 100
        status = "on track" if np_ <= 50 and wp_ <= 30 and sr >= 20 else "off balance"
        insights.append({"kind":"info" if status=="on track" else "warning",
            "title":"50 / 30 / 20 rule",
            "body":f"Needs {np_:.0f}% (≤50%) · Wants {wp_:.0f}% (≤30%) · Savings {sr:.0f}% (≥20%). Status: {status}."})

    # — Savings streak ───────────────────────────────────
    if len(ms) >= 3 and (ms.tail(3)["savings_rate"] >= 20).all():
        insights.append({"kind":"success","title":"3-month savings streak",
            "body":"You've exceeded the 20% savings target for three consecutive months. Consider increasing your SIP allocation."})

    # — Emergency fund ────────────────────────────────────
    avg_exp = ms["expense"].mean()
    target  = avg_exp * 4  # 4 months buffer
    insights.append({"kind":"info","title":"Emergency fund target",
        "body":f"Based on your average monthly expenses (₹{avg_exp:,.0f}), a 4-month emergency fund is ₹{target:,.0f}. Keep this in a liquid instrument."})

    return insights


# ════════════════════════════════════════════════════════════════════════════
#  PLOTLY BASE LAYOUT — all charts inherit this
# ════════════════════════════════════════════════════════════════════════════

def base_layout(**overrides) -> dict:
    # xaxis, yaxis, and legend are intentionally NOT set here.
    # Callers pass them as overrides to avoid "multiple values for keyword
    # argument" crashes when ** unpacking this dict into update_layout().
    # All chart-specific axis/legend config belongs at the call site.
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED, size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(family="Inter", color=WHITE, size=12)),
    )
    layout.update(overrides)
    return layout


# ════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def kpi(label: str, value: str, delta: str = "", delta_kind: str = "flat"):
    """Render a KPI card with Cormorant Garamond number."""
    delta_html = (f'<div class="kpi-delta kpi-{delta_kind}">{delta}</div>'
                  if delta else "")
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def rule(label: str):
    """Gold hairline rule with section label — the signature element."""
    st.markdown(f"""
    <div class="ledger-rule">
      <span class="ledger-rule-label">{label}</span>
      <div class="ledger-rule-line"></div>
    </div>""", unsafe_allow_html=True)


def insight_card(kind: str, title: str, body: str):
    st.markdown(f"""
    <div class="ins-card ins-{kind}">
      <div class="ins-title">{title}</div>
      {body}
    </div>""", unsafe_allow_html=True)


def tip_card(title: str, body: str):
    st.markdown(f"""
    <div class="tip-card">
      <div class="tip-title">{title}</div>
      {body}
    </div>""", unsafe_allow_html=True)


def fmt_inr(v: float) -> str:
    """Format as Indian Rupee with comma grouping."""
    return f"₹{v:,.0f}"


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar contains the add-transaction form.
    Returns updated df if a transaction was saved.
    """
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.4rem 0 1rem;">
          <div style="font-family:'Cormorant Garamond',serif;font-size:1.35rem;
                      letter-spacing:0.2em;text-transform:uppercase;color:{GOLD};">
            Ledger
          </div>
          <div style="font-size:0.68rem;letter-spacing:0.08em;
                      text-transform:uppercase;color:{MUTED};margin-top:0.2rem;">
            Personal Finance
          </div>
        </div>
        <hr style="border:none;border-top:1px solid {BORDER};margin:0 0 1.2rem;">
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:{GOLD};margin-bottom:0.7rem;'>New Transaction</div>", unsafe_allow_html=True)

        t_type   = st.radio("Type", ["expense","income"], horizontal=True)
        amount   = st.number_input("Amount (₹)", min_value=0.01, step=100.0, format="%.2f")
        note     = st.text_input("Description", placeholder="e.g. Zomato order, Monthly salary")
        suggested = auto_categorise(note, t_type)
        cats      = EXPENSE_CATS if t_type == "expense" else INCOME_CATS
        default_i = cats.index(suggested) if suggested in cats else 0
        category  = st.selectbox("Category", cats, index=default_i,
                                  help="Auto-suggested from description")
        tx_date   = st.date_input("Date", value=date.today())

        if st.button("Save Transaction"):
            new_row = pd.DataFrame([{
                "id":       next_id(df),
                "date":     pd.Timestamp(tx_date),
                "type":     t_type,
                "amount":   float(amount),
                "category": category,
                "note":     note,
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            # Write to session state AND disk before rerun so the
            # dashboard reads the updated frame on the very next render.
            st.session_state.df = df
            save_data(df)
            st.success("Saved")
            st.rerun()

        st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:1.2rem 0;'>", unsafe_allow_html=True)

        if not df.empty:
            csv_df = df.copy()
            csv_df["date"] = csv_df["date"].astype(str)
            st.download_button(
                "Export CSV",
                csv_df.to_csv(index=False),
                file_name="ledger_transactions.csv",
                mime="text/csv",
            )

        # ── Stats footer ─────────────────────────────────
        if not df.empty:
            st.markdown(f"""
            <hr style='border:none;border-top:1px solid {BORDER};margin:1.2rem 0;'>
            <div style='font-size:0.7rem;color:{MUTED};line-height:1.9;'>
              <div>{len(df)} transactions recorded</div>
              <div>Since {df["date"].min().strftime("%b %Y")}</div>
            </div>""", unsafe_allow_html=True)

    return df


# ════════════════════════════════════════════════════════════════════════════
#  DEMO DATA — 6 months of realistic transactions
# ════════════════════════════════════════════════════════════════════════════

def generate_demo_data() -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    today = pd.Timestamp.now()
    rows  = []
    tid   = 1
    templates = [
        ("Zomato order",          "Food & Dining",    300,   700),
        ("Swiggy delivery",       "Food & Dining",    250,   600),
        ("Grocery shopping",      "Food & Dining",    900,  2200),
        ("Uber ride",             "Transport",        120,   400),
        ("Petrol",                "Transport",        500,  1400),
        ("Amazon purchase",       "Shopping",         600,  4000),
        ("Myntra order",          "Shopping",        1000,  3500),
        ("Netflix",               "Entertainment",    649,   649),
        ("Spotify",               "Entertainment",    119,   119),
        ("BookMyShow",            "Entertainment",    350,   800),
        ("Pharmacy",              "Health",           200,   900),
        ("Electricity bill",      "Utilities",        600,  1500),
        ("Jio recharge",          "Utilities",        239,   299),
        ("Gym membership",        "Personal Care",    800,   800),
        ("Salon",                 "Personal Care",    300,   600),
        ("Udemy course",          "Education",        499,  1499),
        ("Rent",                  "Rent & Housing", 14000, 14000),
        ("Mutual fund SIP",       "Investment",      3000,  5000),
    ]
    for offset in range(6):
        m_start = (today - pd.DateOffset(months=5 - offset)).replace(day=1)
        # Salary
        rows.append({"id":tid,"date":m_start,"type":"income",
                     "amount":70000.0,"category":"Salary","note":"Monthly salary"})
        tid += 1
        # Occasional freelance
        if offset % 2 == 0:
            rows.append({"id":tid,
                         "date": m_start + pd.Timedelta(days=12),
                         "type":"income",
                         "amount": float(rng.integers(8000, 18000)),
                         "category":"Freelance",
                         "note":"Freelance project"})
            tid += 1
        # Expenses
        chosen = rng.choice(len(templates),
                            size=rng.integers(10, 14), replace=False)
        for idx in chosen:
            note, cat, lo, hi = templates[idx]
            rows.append({
                "id":      tid,
                "date":    m_start + pd.Timedelta(days=int(rng.integers(0, 27))),
                "type":    "expense",
                "amount":  float(rng.integers(lo, hi + 1)),
                "category":cat,
                "note":    note,
            })
            tid += 1
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def tab_dashboard(df: pd.DataFrame):
    now    = pd.Timestamp.now()
    cur_m  = now.to_period("M").strftime("%Y-%m")
    prev_m = (now - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")

    def _m(month): return df[df["date"].dt.to_period("M").astype(str) == month]
    cur, prev = _m(cur_m), _m(prev_m)

    c_inc = cur[cur["type"]=="income"]["amount"].sum()
    c_exp = cur[cur["type"]=="expense"]["amount"].sum()
    c_net = c_inc - c_exp
    c_sr  = (c_net / c_inc * 100) if c_inc > 0 else 0
    p_exp = prev[prev["type"]=="expense"]["amount"].sum()

    # ── KPI strip ────────────────────────────────────────
    rule("This Month")
    k1, k2, k3, k4 = st.columns(4)

    with k1: kpi("Income", fmt_inr(c_inc))
    with k2:
        if p_exp > 0:
            chg = (c_exp - p_exp) / p_exp * 100
            d_label = f"{'▲' if chg>0 else '▼'} {abs(chg):.1f}% vs last month"
            d_kind  = "down" if chg > 0 else "up"
        else:
            d_label, d_kind = "", "flat"
        kpi("Expenses", fmt_inr(c_exp), d_label, d_kind)
    with k3:
        kpi("Net Savings", fmt_inr(c_net),
            f"{'▲' if c_net>=0 else '▼'} Net",
            "up" if c_net >= 0 else "down")
    with k4:
        kpi("Savings Rate", f"{c_sr:.1f}%",
            "Target: 20%",
            "up" if c_sr >= 20 else ("flat" if c_sr >= 10 else "down"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────
    left, right = st.columns([1, 1.9])

    with left:
        rule("Spending Breakdown")
        cat_df = category_spend(df, month=cur_m)
        if not cat_df.empty:
            colours = [CAT_COLOURS.get(c, MUTED) for c in cat_df["category"]]
            fig = go.Figure(go.Pie(
                labels=cat_df["category"],
                values=cat_df["amount"],
                hole=0.60,
                marker=dict(colors=colours, line=dict(color=BLACK, width=2)),
                textinfo="percent",
                textfont=dict(size=10, color=WHITE),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                **base_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    legend=dict(orientation="v", x=1.02, y=0.5,
                                font=dict(size=10, color=MUTED)),
                ),
                annotations=[dict(text=f"<b>{fmt_inr(c_exp)}</b>",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color=WHITE,
                                            family="Cormorant Garamond"))],
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False})
        else:
            st.markdown(f"<div style='color:{MUTED};font-size:0.83rem;padding:2rem 0;'>No expenses recorded this month.</div>", unsafe_allow_html=True)

    with right:
        rule("Income vs Expenses")
        ms = monthly_summary(df)
        if not ms.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Income", x=ms["month"], y=ms["income"],
                marker_color=GREEN, opacity=0.75,
            ))
            fig.add_trace(go.Bar(
                name="Expenses", x=ms["month"], y=ms["expense"],
                marker_color=RED, opacity=0.75,
            ))
            fig.add_trace(go.Scatter(
                name="Net Savings", x=ms["month"], y=ms["net"],
                mode="lines+markers",
                line=dict(color=GOLD, width=2),
                marker=dict(size=6, color=GOLD),
            ))
            fig.update_layout(
                **base_layout(height=300, barmode="group"),
                xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                           tickfont=dict(color=MUTED)),
                yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                           tickfont=dict(color=MUTED)),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False})
        else:
            st.markdown(f"<div style='color:{MUTED};font-size:0.83rem;padding:2rem 0;'>No data yet.</div>", unsafe_allow_html=True)

    # ── Recent transactions ───────────────────────────────
    if not df.empty:
        rule("Recent Transactions")
        recent = df.sort_values("date", ascending=False).head(8).copy()
        recent["date"]   = recent["date"].dt.strftime("%d %b %Y")
        recent["amount"] = recent.apply(
            lambda r: f"+{fmt_inr(r['amount'])}" if r["type"]=="income"
                      else f"-{fmt_inr(r['amount'])}", axis=1)
        st.dataframe(
            recent[["date","category","note","amount"]].rename(columns={
                "date":"Date","category":"Category",
                "note":"Description","amount":"Amount"}),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def tab_analysis(df: pd.DataFrame):
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No data. Load demo data or add transactions.</div>", unsafe_allow_html=True)
        return

    months    = sorted(df["date"].dt.to_period("M").astype(str).unique(), reverse=True)
    all_label = "All time"
    sel_month = st.selectbox("Period", [all_label] + list(months),
                              label_visibility="collapsed")
    filt_month = None if sel_month == all_label else sel_month
    cat_df     = category_spend(df, month=filt_month)

    # ── Bar + treemap ─────────────────────────────────────
    rule("Category Breakdown")
    left, right = st.columns(2)

    with left:
        if not cat_df.empty:
            fig = px.bar(
                cat_df, x="amount", y="category", orientation="h",
                color="category", color_discrete_map=CAT_COLOURS,
                labels={"amount":"","category":""},
                text=cat_df["amount"].apply(fmt_inr),
            )
            fig.update_traces(textposition="outside",
                              textfont=dict(size=10, color=MUTED))
            fig.update_layout(
                **base_layout(height=380, showlegend=False),
                yaxis=dict(categoryorder="total ascending",
                           gridcolor=BORDER, linecolor=BORDER,
                           tickfont=dict(color=MUTED)),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False})

    with right:
        if not cat_df.empty:
            fig = px.treemap(
                cat_df, path=["category"], values="amount",
                color="amount",
                color_continuous_scale=[[0,CARD],[0.5,GOLD_DIM],[1,GOLD]],
            )
            fig.update_layout(**base_layout(height=380,
                              margin=dict(l=0,r=0,t=0,b=0)))
            fig.update_traces(
                textfont=dict(family="Inter", color=BLACK, size=11),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False})

    # ── Category MoM heatmap ──────────────────────────────
    rule("Month-over-Month Heatmap")
    pivot = df[df["type"]=="expense"].copy()
    pivot["month"] = pivot["date"].dt.to_period("M").astype(str)
    heat  = pivot.pivot_table(index="category", columns="month",
                              values="amount", aggfunc="sum", fill_value=0)
    if not heat.empty:
        fig = go.Figure(go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[[0,BLACK],[0.3,GOLD_DIM],[1,GOLD]],
            hovertemplate="<b>%{y}</b><br>%{x}<br>₹%{z:,.0f}<extra></extra>",
            showscale=True,
        ))
        fig.update_layout(
            **base_layout(height=340,
                          margin=dict(l=140, r=20, t=20, b=20)),
            xaxis=dict(tickfont=dict(color=MUTED, size=10)),
            yaxis=dict(tickfont=dict(color=MUTED, size=10)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})

    # ── Savings rate bars ─────────────────────────────────
    rule("Savings Rate by Month")
    ms = monthly_summary(df)
    if not ms.empty:
        colours = [GREEN if v >= 20 else (GOLD if v >= 0 else RED)
                   for v in ms["savings_rate"]]
        fig = go.Figure(go.Bar(
            x=ms["month"], y=ms["savings_rate"],
            marker_color=colours,
            text=[f"{v:.1f}%" for v in ms["savings_rate"]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
        ))
        fig.add_hline(y=20, line_dash="dot", line_color=GOLD_DIM,
                      annotation_text="20% target",
                      annotation_font_color=GOLD_DIM, annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=260),
            yaxis=dict(title="", gridcolor=BORDER, linecolor=BORDER,
                       ticksuffix="%", tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CASH FLOW
# ════════════════════════════════════════════════════════════════════════════

def tab_cashflow(df: pd.DataFrame):
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No data yet.</div>", unsafe_allow_html=True)
        return

    months    = sorted(df["date"].dt.to_period("M").astype(str).unique(), reverse=True)
    sel_month = st.selectbox("Month", months, label_visibility="collapsed")

    # ── Waterfall chart ───────────────────────────────────
    rule("Cash Flow Waterfall")
    month_df  = df[df["date"].dt.to_period("M").astype(str) == sel_month]
    inc_total = month_df[month_df["type"]=="income"]["amount"].sum()
    exp_cats  = (month_df[month_df["type"]=="expense"]
                   .groupby("category")["amount"].sum()
                   .sort_values(ascending=False))

    if not exp_cats.empty:
        labels  = ["Income"] + list(exp_cats.index) + ["Net"]
        values  = [inc_total] + [-v for v in exp_cats.values] + [0]
        net_val = inc_total - exp_cats.sum()
        values[-1] = net_val

        measures = (["absolute"] +
                    ["relative"] * len(exp_cats) +
                    ["total"])
        bar_colours = (
            [GREEN]
            + [RED] * len(exp_cats)
            + [GREEN if net_val >= 0 else RED]
        )
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color=BORDER, width=1)),
            increasing=dict(marker_color=GREEN),
            decreasing=dict(marker_color=RED),
            totals=dict(marker_color=GOLD),
            textposition="outside",
            text=[fmt_inr(abs(v)) for v in values],
            textfont=dict(size=10, color=MUTED),
        ))
        fig.update_layout(
            **base_layout(height=360, showlegend=False),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                       tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})

    # ── Daily spend ───────────────────────────────────────
    rule("Daily Spending Pattern")
    daily = daily_spend(df, sel_month)
    if not daily.empty:
        avg_daily = daily["amount"].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["amount"],
            marker_color=GOLD_DIM, name="Daily spend",
        ))
        fig.add_hline(y=avg_daily, line_dash="dot", line_color=GOLD,
                      annotation_text=f"Avg {fmt_inr(avg_daily)}/day",
                      annotation_font_color=GOLD, annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=240),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                       tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                       tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})

    # ── Cumulative spend ──────────────────────────────────
    rule("Cumulative Spend vs Income")
    month_exp = (df[(df["type"]=="expense") &
                    (df["date"].dt.to_period("M").astype(str) == sel_month)]
                   .sort_values("date").copy())
    if not month_exp.empty:
        month_exp["cumulative"] = month_exp["amount"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=month_exp["date"], y=month_exp["cumulative"],
            mode="lines", name="Cumulative spend",
            line=dict(color=RED, width=2),
            fill="tozeroy", fillcolor=f"rgba(207,102,121,0.08)",
        ))
        fig.add_hline(y=inc_total, line_dash="dot", line_color=GREEN,
                      annotation_text=f"Income {fmt_inr(inc_total)}",
                      annotation_font_color=GREEN, annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=220),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                       tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                       tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════

def tab_predictions(df: pd.DataFrame):
    ms = monthly_summary(df)
    if len(ms) < 3:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>Predictions require at least 3 months of data. You have {len(ms)} so far — keep logging.</div>", unsafe_allow_html=True)
        return

    preds = predict_expenses(df, n=3)

    # ── Forecast KPIs ─────────────────────────────────────
    rule("3-Month Expense Forecast")
    if not preds.empty:
        avg_income = ms["income"].mean()
        cols = st.columns(3)
        for i, (_, row) in enumerate(preds.iterrows()):
            proj_net = avg_income - row["predicted"]
            proj_sr  = (proj_net / avg_income * 100) if avg_income > 0 else 0
            diff     = row["predicted"] - ms["expense"].mean()
            with cols[i]:
                kpi(row["month"],
                    fmt_inr(row["predicted"]),
                    f"{'▲' if diff>0 else '▼'} {fmt_inr(abs(diff))} vs avg",
                    "down" if diff > 0 else "up")

    # ── Forecast + historical line chart ──────────────────
    rule("Trend & Forecast")
    fig = go.Figure()

    # Confidence band
    if not preds.empty:
        fig.add_trace(go.Scatter(
            x=list(preds["month"]) + list(reversed(preds["month"])),
            y=list(preds["upper"]) + list(reversed(preds["lower"])),
            fill="toself",
            fillcolor=f"rgba(201,168,76,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence band",
            hoverinfo="skip",
        ))

    # Historical
    fig.add_trace(go.Scatter(
        x=ms["month"], y=ms["expense"],
        mode="lines+markers", name="Actual expenses",
        line=dict(color=RED, width=2),
        marker=dict(size=6, color=RED),
    ))
    fig.add_trace(go.Scatter(
        x=ms["month"], y=ms["income"],
        mode="lines+markers", name="Income",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6, color=GREEN),
    ))

    # Bridge + forecast
    if not preds.empty:
        bridge_x = [ms["month"].iloc[-1]] + list(preds["month"])
        bridge_y = [ms["expense"].iloc[-1]] + list(preds["predicted"])
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode="lines+markers", name="Forecast",
            line=dict(color=GOLD, width=2, dash="dot"),
            marker=dict(size=7, color=GOLD, symbol="diamond"),
        ))

    fig.update_layout(
        **base_layout(height=360),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar":False})

    # ── Projected savings ─────────────────────────────────
    if not preds.empty:
        rule("Projected Savings (at avg income)")
        avg_income = ms["income"].mean()
        s_cols = st.columns(3)
        for i, (_, row) in enumerate(preds.iterrows()):
            proj_net = avg_income - row["predicted"]
            proj_sr  = (proj_net / avg_income * 100) if avg_income > 0 else 0
            with s_cols[i]:
                kpi(f"{row['month']} savings",
                    fmt_inr(proj_net),
                    f"{proj_sr:.1f}% savings rate",
                    "up" if proj_sr >= 20 else "down")

    st.markdown(f"""
    <div style='font-size:0.75rem;color:{MUTED};margin-top:1.5rem;padding:0.85rem 1rem;
    background:{CARD};border-left:2px solid {BORDER};'>
    Forecast uses linear regression on monthly expense totals. Accuracy improves
    with more data. Treat projections as directional indicators, not guarantees.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

def tab_insights(df: pd.DataFrame):
    rule("Financial Insights")
    for ins in generate_insights(df):
        insight_card(ins["kind"], ins["title"], ins["body"])

    if df.empty:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    rule("Financial Principles")

    ms = monthly_summary(df)
    avg_exp = ms["expense"].mean() if not ms.empty else 0

    tips = [
        ("50 / 30 / 20 rule",
         "Allocate 50% of income to needs (rent, groceries, utilities), 30% to wants (dining, entertainment, shopping), and 20% to savings and investments. Revisit these ratios every quarter."),
        ("Pay yourself first",
         "Move your savings target to a separate account on salary day — before any discretionary spending. Automating this removes willpower from the equation."),
        ("Emergency fund",
         f"Your 4-month buffer target is approximately {fmt_inr(avg_exp * 4)}. Hold it in a liquid instrument: high-interest savings account or liquid mutual fund."),
        ("Index fund SIP",
         "A monthly SIP of ₹5,000 into a broad market index fund at 12% CAGR becomes approximately ₹50 lakh over 20 years. Starting early compresses the time needed dramatically."),
        ("Subscription audit",
         "List every recurring charge and cancel any you haven't actively used in the past 30 days. People routinely discover ₹1,000–2,000/month in forgotten subscriptions."),
        ("Lifestyle inflation",
         "When income rises, resist the pressure to scale expenses proportionally. A salary hike is most powerful when it goes entirely into investments, not lifestyle upgrades."),
    ]
    cols = st.columns(2)
    for i, (title, body) in enumerate(tips):
        with cols[i % 2]:
            tip_card(title, body)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — TRANSACTIONS
# ════════════════════════════════════════════════════════════════════════════

def tab_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No transactions recorded yet.</div>", unsafe_allow_html=True)
        return df

    # ── Filter bar ────────────────────────────────────────
    rule("Filter")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        t_filter = st.selectbox("Type", ["All","income","expense"],
                                label_visibility="visible")
    with fc2:
        months = ["All"] + sorted(
            df["date"].dt.to_period("M").astype(str).unique(), reverse=True)
        m_filter = st.selectbox("Month", months, label_visibility="visible")
    with fc3:
        cats = ["All"] + sorted(df["category"].unique())
        c_filter = st.selectbox("Category", cats, label_visibility="visible")
    with fc4:
        search = st.text_input("Search description", placeholder="keyword…",
                               label_visibility="visible")

    filtered = df.copy()
    if t_filter != "All":
        filtered = filtered[filtered["type"] == t_filter]
    if m_filter != "All":
        filtered = filtered[
            filtered["date"].dt.to_period("M").astype(str) == m_filter]
    if c_filter != "All":
        filtered = filtered[filtered["category"] == c_filter]
    if search:
        filtered = filtered[
            filtered["note"].str.contains(search, case=False, na=False)]

    # ── Summary strip ─────────────────────────────────────
    rule("Summary")
    s1, s2, s3 = st.columns(3)
    with s1: kpi("Transactions", str(len(filtered)))
    with s2: kpi("Total Income",
                 fmt_inr(filtered[filtered["type"]=="income"]["amount"].sum()))
    with s3: kpi("Total Expenses",
                 fmt_inr(filtered[filtered["type"]=="expense"]["amount"].sum()))

    # ── Table ─────────────────────────────────────────────
    rule("Ledger")
    display = filtered.sort_values("date", ascending=False).copy()
    display["date"]   = display["date"].dt.strftime("%d %b %Y")
    display["amount"] = display.apply(
        lambda r: (f"+{fmt_inr(r['amount'])}"
                   if r["type"]=="income" else f"-{fmt_inr(r['amount'])}"), axis=1)
    st.dataframe(
        display[["id","date","type","category","note","amount"]].rename(columns={
            "id":"#","date":"Date","type":"Type",
            "category":"Category","note":"Description","amount":"Amount"}),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.caption(f"{len(filtered)} of {len(df)} transactions")

    # ── Delete ────────────────────────────────────────────
    rule("Delete Transaction")
    d1, d2 = st.columns([1, 3])
    with d1:
        del_id = st.number_input("Transaction #", min_value=1, step=1,
                                  label_visibility="visible")
    with d2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Delete"):
            before = len(df)
            df = df[df["id"] != del_id].reset_index(drop=True)
            if len(df) < before:
                st.session_state.df = df
                save_data(df)
                st.success(f"Transaction #{del_id} deleted.")
                st.rerun()
            else:
                st.error("ID not found.")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if "df" not in st.session_state:
        st.session_state.df = load_data()

    st.session_state.df = render_sidebar(st.session_state.df)
    df = st.session_state.df

    # ── Page header ───────────────────────────────────────
    st.markdown(f"""
    <div class="page-header">
      <span class="page-wordmark">Ledger</span>
      <span class="page-sub">Personal Finance Intelligence</span>
    </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────
    tabs = st.tabs([
        "Dashboard",
        "Analysis",
        "Cash Flow",
        "Predictions",
        "Insights",
        "Transactions",
    ])

    with tabs[0]: tab_dashboard(df)
    with tabs[1]: tab_analysis(df)
    with tabs[2]: tab_cashflow(df)
    with tabs[3]: tab_predictions(df)
    with tabs[4]: tab_insights(df)
    with tabs[5]:
        updated = tab_transactions(df)
        st.session_state.df = updated


if __name__ == "__main__":
    main()