"""
LEDGER — Smart Personal Finance Tracker
Multi-user edition with Supabase Auth + Postgres

Each user has their own isolated data. Auth flow:
  Sign Up → email confirmation → Sign In → dashboard
  (or Sign In with existing account)

Setup
-----
1. pip install streamlit pandas numpy plotly scikit-learn supabase
2. Create a .streamlit/secrets.toml  (see SETUP.md)
3. streamlit run app.py
"""

# ── stdlib ───────────────────────────────────────────────
import os
from datetime import date

# ── third-party ──────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client
from supabase.client import AuthApiError   # supabase 2.x (via supabase_auth)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG — first Streamlit call
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
GOLD       = "#C9A84C"
GOLD_DIM   = "#8A6E2F"
GOLD_LIGHT = "#E8D5A3"
BLACK      = "#0A0A0A"
CARD       = "#111111"
BORDER     = "#1E1E1E"
MUTED      = "#6B6B6B"
WHITE      = "#F5F5F0"
GREEN      = "#4CAF7D"
RED        = "#CF6679"

CAT_COLOURS = {
    "Food & Dining":  "#C9A84C", "Transport":    "#8A9BB5",
    "Shopping":       "#A89BC0", "Entertainment":"#7EB5A6",
    "Health":         "#7DAF7D", "Utilities":    "#B5A97E",
    "Rent & Housing": "#C09080", "Education":    "#8AABB5",
    "Personal Care":  "#B58A9B", "Investment":   "#9BB58A",
    "Salary":         "#4CAF7D", "Freelance":    "#6B9FBF",
    "Other Income":   "#9FAFBF", "Miscellaneous":"#6B6B6B",
}

# ════════════════════════════════════════════════════════════════════════════
#  CATEGORIES
# ════════════════════════════════════════════════════════════════════════════
CATEGORIES = {
    "Food & Dining":    ["zomato","swiggy","restaurant","food","cafe","coffee",
                         "lunch","dinner","breakfast","grocery","supermarket",
                         "blinkit","zepto","hotel","pizza","burger"],
    "Transport":        ["uber","ola","petrol","fuel","metro","bus","auto",
                         "cab","parking","rapido","irctc","train","flight"],
    "Shopping":         ["amazon","flipkart","myntra","ajio","mall","shop",
                         "meesho","nykaa","purchase","buy","store"],
    "Entertainment":    ["netflix","prime","spotify","hotstar","disney",
                         "movie","game","youtube","bookmyshow","concert"],
    "Health":           ["pharmacy","doctor","hospital","clinic","medicine",
                         "apollo","medplus","healthkart","lab","test"],
    "Utilities":        ["electricity","water","gas","internet","wifi",
                         "broadband","recharge","dth","jio","airtel","vi"],
    "Rent & Housing":   ["rent","maintenance","housing","apartment","society",
                         "landlord","deposit","lease"],
    "Education":        ["course","udemy","coursera","book","tuition","school",
                         "college","fee","udacity","skillshare"],
    "Personal Care":    ["salon","haircut","spa","gym","fitness","grooming",
                         "parlour","laundry"],
    "Investment":       ["mutual fund","sip","stocks","zerodha","groww",
                         "crypto","fd","nps","ppf","smallcase","etf"],
    "Salary":           ["salary","payroll","ctc","stipend","wages"],
    "Freelance":        ["freelance","project","client","invoice",
                         "payment received","transfer received","consulting"],
    "Other Income":     ["cashback","refund","gift","bonus","interest",
                         "dividend","award"],
    "Miscellaneous":    [],
}

EXPENSE_CATS = [c for c in CATEGORIES
                if c not in ("Salary", "Freelance", "Other Income")]
INCOME_CATS  = ["Salary", "Freelance", "Other Income"]

# ════════════════════════════════════════════════════════════════════════════
#  SUPABASE CLIENT  (reads from .streamlit/secrets.toml)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    """
    Returns a cached Supabase client.
    Reads SUPABASE_URL and SUPABASE_KEY from Streamlit secrets.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def sign_up(email: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    try:
        sb = get_supabase()
        sb.auth.sign_up({"email": email, "password": password})
        return True, "Account created. Check your email to confirm, then sign in."
    except AuthApiError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Sign in and store session in session_state. Returns (success, message)."""
    try:
        sb = get_supabase()
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user    = res.user
        st.session_state.session = res.session
        return True, "Signed in."
    except AuthApiError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def sign_out():
    """Sign out and clear session state."""
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    for key in ("user", "session", "df"):
        st.session_state.pop(key, None)
    st.rerun()


def current_user_id() -> str | None:
    """Return the UUID of the logged-in user, or None."""
    user = st.session_state.get("user")
    return user.id if user else None


def current_user_email() -> str:
    user = st.session_state.get("user")
    return user.email if user else ""


# ════════════════════════════════════════════════════════════════════════════
#  DATABASE LAYER  — all queries scoped to the current user's UUID
#  Row-Level Security (RLS) on Supabase ensures the DB also enforces this.
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Fetch all transactions for the current user from Supabase.
    Returns a DataFrame sorted by date descending.
    """
    uid = current_user_id()
    if not uid:
        return _empty_df()
    try:
        sb  = get_supabase()
        res = sb.table("transactions") \
                .select("*") \
                .eq("user_id", uid) \
                .order("date", desc=True) \
                .execute()
        rows = res.data
        if not rows:
            return _empty_df()
        df = pd.DataFrame(rows)
        df["date"]   = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return _empty_df()


def _empty_df() -> pd.DataFrame:
    """
    Return an empty DataFrame with correct dtypes.
    date must be datetime64 so .dt accessors never raise
    'Can only use .dt accessor with datetimelike values'.
    """
    df = pd.DataFrame(
        columns=["id", "user_id", "date", "type", "amount", "category", "note"])
    df["date"]   = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the date column to datetime64 (no-op if already correct).
    Called at the top of every tab as a safeguard against Supabase
    returning date strings instead of parsed timestamps.
    """
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def insert_transaction(t_type: str, amount: float,
                       category: str, note: str, tx_date: date) -> bool:
    """Insert one transaction row for the current user. Returns success bool."""
    uid = current_user_id()
    if not uid:
        return False
    try:
        get_supabase().table("transactions").insert({
            "user_id":  uid,
            "date":     tx_date.isoformat(),
            "type":     t_type,
            "amount":   amount,
            "category": category,
            "note":     note,
        }).execute()
        return True
    except Exception as e:
        st.error(f"Insert failed: {e}")
        return False


def delete_transaction_db(row_id: int) -> bool:
    """Delete a transaction row only if it belongs to the current user."""
    uid = current_user_id()
    if not uid:
        return False
    try:
        res = get_supabase().table("transactions") \
                  .delete() \
                  .eq("id", row_id) \
                  .eq("user_id", uid) \
                  .execute()
        return len(res.data) > 0
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  AUTO-CATEGORISATION
# ════════════════════════════════════════════════════════════════════════════

def auto_categorise(note: str, t_type: str) -> str:
    text = (note or "").lower()
    best_cat, best_score = "Miscellaneous", 0
    pool = INCOME_CATS if t_type == "income" else EXPENSE_CATS
    for cat in pool:
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
    df = ensure_datetime(df)
    if df.empty:
        return pd.DataFrame(columns=["month","income","expense","net","savings_rate"])
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M")
    g = tmp.groupby(["month","type"])["amount"].sum().unstack(fill_value=0)
    g.columns.name = None
    for col in ("income","expense"):
        if col not in g.columns:
            g[col] = 0.0
    g = g[["income","expense"]].reset_index()
    g["month"] = g["month"].astype(str)
    g["net"]   = g["income"] - g["expense"]
    g["savings_rate"] = np.where(
        g["income"] > 0, (g["net"] / g["income"] * 100).round(1), 0.0)
    return g.sort_values("month").reset_index(drop=True)


def category_spend(df: pd.DataFrame, month: str = None) -> pd.DataFrame:
    df = ensure_datetime(df)
    tmp = df[df["type"] == "expense"].copy()
    if month:
        tmp = tmp[tmp["date"].dt.to_period("M").astype(str) == month]
    if tmp.empty:
        return pd.DataFrame(columns=["category","amount"])
    return tmp.groupby("category")["amount"].sum().reset_index() \
              .sort_values("amount", ascending=False)


def daily_spend(df: pd.DataFrame, month: str) -> pd.DataFrame:
    df = ensure_datetime(df)
    tmp = df[(df["type"]=="expense") &
             (df["date"].dt.to_period("M").astype(str)==month)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["date","amount"])
    return tmp.groupby("date")["amount"].sum().reset_index().sort_values("date")


def predict_expenses(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    df = ensure_datetime(df)
    ms = monthly_summary(df)
    if len(ms) < 3:
        return pd.DataFrame()
    ms = ms.reset_index(drop=True)
    ms["x"] = np.arange(len(ms))
    X, y  = ms[["x"]].values, ms["expense"].values
    model = LinearRegression().fit(X, y)
    std   = (y - model.predict(X)).std()
    last_x, last_p = ms["x"].max(), pd.Period(ms["month"].iloc[-1], "M")
    rows = []
    for i in range(1, n+1):
        pred = max(model.predict([[last_x+i]])[0], 0)
        rows.append({"month":(last_p+i).strftime("%Y-%m"),
                     "predicted":round(pred,2),
                     "lower":round(max(pred-std,0),2),
                     "upper":round(pred+std,2)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHT ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> list:
    df = ensure_datetime(df)
    if df.empty:
        return [{"kind":"info","title":"Getting started",
                 "body":"Add your first transaction. Insights appear as your history builds."}]
    ms     = monthly_summary(df)
    now    = pd.Timestamp.now()
    cur_m  = now.to_period("M").strftime("%Y-%m")
    prev_m = (now - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")
    cur    = ms[ms["month"]==cur_m]
    prev   = ms[ms["month"]==prev_m]
    def _v(row, col): return row[col].values[0] if not row.empty else 0.0
    income = _v(cur,"income"); expense = _v(cur,"expense")
    net    = _v(cur,"net");    sr      = _v(cur,"savings_rate")
    p_exp  = _v(prev,"expense")
    cat_df = category_spend(df, month=cur_m)
    ins    = []

    if income > 0:
        if sr >= 30:
            ins.append({"kind":"success","title":"Strong savings rate",
                "body":f"You saved {sr:.1f}% of income this month — well above the 20% benchmark."})
        elif sr >= 20:
            ins.append({"kind":"success","title":"On target",
                "body":f"Savings rate: {sr:.1f}%. Consider routing the surplus into an index fund SIP."})
        elif sr >= 0:
            gap = (0.20*income) - net
            ins.append({"kind":"warning","title":"Below savings target",
                "body":f"Savings rate: {sr:.1f}%. Cutting ₹{gap:,.0f} in discretionary spending would hit 20%."})
        else:
            ins.append({"kind":"danger","title":"Spending exceeds income",
                "body":f"Deficit of ₹{abs(net):,.0f} this month. Review your top three expense categories."})

    if p_exp > 0:
        chg = (expense - p_exp) / p_exp * 100
        if chg > 20:
            ins.append({"kind":"danger","title":"Expenses spiked",
                "body":f"Spending up {chg:.1f}% vs last month (₹{p_exp:,.0f} → ₹{expense:,.0f})."})
        elif chg < -10:
            ins.append({"kind":"success","title":"Spending reduced",
                "body":f"Expenses fell {abs(chg):.1f}% compared to last month."})

    if not cat_df.empty and income > 0:
        top_cat = cat_df.iloc[0]["category"]
        top_amt = cat_df.iloc[0]["amount"]
        top_pct = top_amt / income * 100
        if top_pct > 35:
            ins.append({"kind":"warning","title":f"Heavy spend on {top_cat}",
                "body":f"{top_cat} is consuming {top_pct:.1f}% of your income (₹{top_amt:,.0f})."})

    food = cat_df[cat_df["category"]=="Food & Dining"]["amount"].sum() if not cat_df.empty else 0
    if income > 0 and food/income > 0.25:
        ins.append({"kind":"warning","title":"Food spend is high",
            "body":f"Food & Dining is {food/income*100:.1f}% of income. Meal prepping typically cuts this 25–35%."})

    ent = cat_df[cat_df["category"]=="Entertainment"]["amount"].sum() if not cat_df.empty else 0
    if income > 0 and ent/income > 0.12:
        ins.append({"kind":"info","title":"Review subscriptions",
            "body":f"Entertainment is {ent/income*100:.1f}% of income. Audit unused streaming services."})

    if income > 0:
        needs = cat_df[cat_df["category"].isin(
            ["Rent & Housing","Utilities","Food & Dining","Health","Transport"]
        )]["amount"].sum() if not cat_df.empty else 0
        wants = cat_df[cat_df["category"].isin(
            ["Entertainment","Shopping","Personal Care"]
        )]["amount"].sum() if not cat_df.empty else 0
        np_ = needs/income*100; wp_ = wants/income*100
        ok  = np_ <= 50 and wp_ <= 30 and sr >= 20
        ins.append({"kind":"info" if ok else "warning","title":"50 / 30 / 20 rule",
            "body":f"Needs {np_:.0f}% (≤50%) · Wants {wp_:.0f}% (≤30%) · Savings {sr:.0f}% (≥20%). {'On track.' if ok else 'Off balance.'}"})

    if len(ms) >= 3 and (ms.tail(3)["savings_rate"] >= 20).all():
        ins.append({"kind":"success","title":"3-month savings streak",
            "body":"You've hit the 20% savings target three months running."})

    avg_exp = ms["expense"].mean()
    ins.append({"kind":"info","title":"Emergency fund target",
        "body":f"4-month buffer at your spending level: ₹{avg_exp*4:,.0f}. Keep it in a liquid instrument."})

    return ins


# ════════════════════════════════════════════════════════════════════════════
#  PLOTLY BASE LAYOUT
# ════════════════════════════════════════════════════════════════════════════

def base_layout(**overrides) -> dict:
    # xaxis / yaxis / legend intentionally excluded — callers pass them
    # directly to update_layout() to avoid duplicate-keyword-argument errors.
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
#  CSS
# ════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;color:{WHITE};}}
.stApp{{background:{BLACK};}}
.block-container{{padding:0 2.5rem 3rem;max-width:1400px;}}

/* Sidebar */
section[data-testid="stSidebar"]{{background:{CARD};border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label span{{color:{WHITE}!important;font-size:0.82rem;}}
section[data-testid="stSidebar"] .stButton>button{{
  background:transparent;border:1px solid {GOLD};color:{GOLD}!important;
  border-radius:2px;font-size:0.82rem;font-weight:500;letter-spacing:0.06em;
  text-transform:uppercase;padding:0.5rem 1rem;width:100%;transition:all 0.2s;}}
section[data-testid="stSidebar"] .stButton>button:hover{{background:{GOLD};color:{BLACK}!important;}}

/* Buttons */
.main .stButton>button{{
  background:transparent;border:1px solid {GOLD};color:{GOLD}!important;
  border-radius:2px;font-size:0.8rem;font-weight:500;letter-spacing:0.06em;
  text-transform:uppercase;padding:0.45rem 1.2rem;transition:all 0.2s;}}
.main .stButton>button:hover{{background:{GOLD};color:{BLACK}!important;}}

/* Auth form submit button — full gold fill */
.auth-submit .stButton>button{{
  background:{GOLD}!important;color:{BLACK}!important;
  border:none;width:100%;font-size:0.9rem;padding:0.65rem;}}
.auth-submit .stButton>button:hover{{background:{GOLD_LIGHT}!important;}}

/* Inputs */
input,textarea,select,
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div{{
  background:{BLACK}!important;border-color:{BORDER}!important;
  color:{WHITE}!important;border-radius:2px!important;font-size:0.85rem!important;}}
input:focus{{border-color:{GOLD}!important;box-shadow:0 0 0 1px {GOLD}!important;}}
label{{color:{MUTED}!important;font-size:0.75rem!important;
       letter-spacing:0.05em;text-transform:uppercase;}}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:transparent;border-bottom:1px solid {BORDER};gap:0;}}
.stTabs [data-baseweb="tab"]{{
  background:transparent;color:{MUTED};border-radius:0;
  border-bottom:2px solid transparent;font-size:0.8rem;font-weight:500;
  letter-spacing:0.07em;text-transform:uppercase;padding:0.7rem 1.4rem;margin-bottom:-1px;}}
.stTabs [aria-selected="true"]{{color:{GOLD}!important;border-bottom:2px solid {GOLD}!important;background:transparent!important;}}

/* KPI card */
.kpi-card{{background:{CARD};border:1px solid {BORDER};border-top:2px solid {GOLD};padding:1.4rem 1.6rem 1.2rem;}}
.kpi-label{{font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};margin-bottom:0.5rem;}}
.kpi-value{{font-family:'Cormorant Garamond',Georgia,serif;font-size:2.1rem;font-weight:400;color:{WHITE};line-height:1;}}
.kpi-delta{{font-size:0.73rem;font-weight:500;margin-top:0.4rem;letter-spacing:0.02em;}}
.kpi-up{{color:{GREEN};}}.kpi-down{{color:{RED};}}.kpi-flat{{color:{MUTED};}}

/* Section rule */
.ledger-rule{{display:flex;align-items:center;gap:0.75rem;margin:1.8rem 0 1.1rem;}}
.ledger-rule-label{{font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{GOLD};white-space:nowrap;}}
.ledger-rule-line{{flex:1;height:1px;background:{BORDER};}}

/* Insight */
.ins-card{{border-left:2px solid;background:{CARD};padding:0.85rem 1.1rem;margin-bottom:0.55rem;font-size:0.85rem;color:{WHITE};line-height:1.6;border-radius:0 2px 2px 0;}}
.ins-success{{border-color:{GREEN};}}.ins-warning{{border-color:{GOLD};}}.ins-danger{{border-color:{RED};}}.ins-info{{border-color:#8A9BB5;}}
.ins-title{{font-weight:600;color:{GOLD_LIGHT};margin-bottom:0.2rem;font-size:0.8rem;letter-spacing:0.04em;text-transform:uppercase;}}

/* Tip */
.tip-card{{background:{CARD};border:1px solid {BORDER};padding:1rem 1.2rem;margin-bottom:0.5rem;font-size:0.83rem;color:{WHITE};line-height:1.6;}}
.tip-title{{font-weight:600;color:{GOLD};font-size:0.78rem;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.3rem;}}

/* Auth card */
.auth-card{{
  background:{CARD};border:1px solid {BORDER};border-top:2px solid {GOLD};
  max-width:420px;margin:4rem auto 0;padding:2.5rem 2.5rem 2rem;}}
.auth-wordmark{{font-family:'Cormorant Garamond',serif;font-size:1.6rem;
  letter-spacing:0.22em;text-transform:uppercase;color:{GOLD};
  text-align:center;margin-bottom:0.25rem;}}
.auth-sub{{font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
  color:{MUTED};text-align:center;margin-bottom:2rem;}}
.auth-error{{background:rgba(207,102,121,0.12);border:1px solid {RED};
  color:{RED};padding:0.65rem 0.9rem;font-size:0.82rem;margin-bottom:1rem;border-radius:2px;}}
.auth-success{{background:rgba(76,175,125,0.12);border:1px solid {GREEN};
  color:{GREEN};padding:0.65rem 0.9rem;font-size:0.82rem;margin-bottom:1rem;border-radius:2px;}}

/* Page header */
.page-header{{display:flex;align-items:baseline;gap:1rem;padding:2rem 0 0.5rem;border-bottom:1px solid {BORDER};margin-bottom:0.5rem;}}
.page-wordmark{{font-family:'Cormorant Garamond',serif;font-size:1.55rem;font-weight:400;letter-spacing:0.18em;text-transform:uppercase;color:{GOLD};}}
.page-sub{{font-size:0.73rem;color:{MUTED};letter-spacing:0.06em;text-transform:uppercase;}}

/* Scrollbar */
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{BLACK};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:2px;}}
::-webkit-scrollbar-thumb:hover{{background:{GOLD_DIM};}}
</style>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════════════════

def kpi(label, value, delta="", delta_kind="flat"):
    delta_html = f'<div class="kpi-delta kpi-{delta_kind}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def rule(label):
    st.markdown(f"""
    <div class="ledger-rule">
      <span class="ledger-rule-label">{label}</span>
      <div class="ledger-rule-line"></div>
    </div>""", unsafe_allow_html=True)


def insight_card(kind, title, body):
    st.markdown(f'<div class="ins-card ins-{kind}"><div class="ins-title">{title}</div>{body}</div>',
                unsafe_allow_html=True)


def tip_card(title, body):
    st.markdown(f'<div class="tip-card"><div class="tip-title">{title}</div>{body}</div>',
                unsafe_allow_html=True)


def fmt_inr(v): return f"₹{v:,.0f}"


# ════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE  — shown when no user is logged in
# ════════════════════════════════════════════════════════════════════════════

def render_auth_page():
    """Full-page sign-in / sign-up form. Stops execution if not authenticated."""
    st.markdown("""
    <div class="auth-card">
      <div class="auth-wordmark">Ledger</div>
      <div class="auth-sub">Personal Finance Intelligence</div>
    </div>""", unsafe_allow_html=True)

    # Centre the form by putting it inside the same max-width card via columns
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        mode = st.radio("", ["Sign In", "Sign Up"],
                        horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if mode == "Sign Up":
            confirm = st.text_input("Confirm password", type="password")

        st.markdown("<br>", unsafe_allow_html=True)

        # Feedback placeholders
        msg_slot = st.empty()

        st.markdown('<div class="auth-submit">', unsafe_allow_html=True)
        submitted = st.button(mode, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            if not email or not password:
                msg_slot.markdown(
                    '<div class="auth-error">Email and password are required.</div>',
                    unsafe_allow_html=True)
            elif mode == "Sign Up":
                if password != confirm:
                    msg_slot.markdown(
                        '<div class="auth-error">Passwords do not match.</div>',
                        unsafe_allow_html=True)
                elif len(password) < 6:
                    msg_slot.markdown(
                        '<div class="auth-error">Password must be at least 6 characters.</div>',
                        unsafe_allow_html=True)
                else:
                    ok, msg = sign_up(email, password)
                    cls = "auth-success" if ok else "auth-error"
                    msg_slot.markdown(
                        f'<div class="{cls}">{msg}</div>',
                        unsafe_allow_html=True)
            else:  # Sign In
                ok, msg = sign_in(email, password)
                if ok:
                    st.session_state.df = load_data()
                    st.rerun()
                else:
                    msg_slot.markdown(
                        f'<div class="auth-error">{msg}</div>',
                        unsafe_allow_html=True)

    st.stop()  # Don't render the dashboard until authenticated


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  — transaction form + user info
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.4rem 0 1rem;">
          <div style="font-family:'Cormorant Garamond',serif;font-size:1.35rem;
                      letter-spacing:0.2em;text-transform:uppercase;color:{GOLD};">
            Ledger
          </div>
          <div style="font-size:0.68rem;letter-spacing:0.08em;
                      text-transform:uppercase;color:{MUTED};margin-top:0.2rem;">
            {current_user_email()}
          </div>
        </div>
        <hr style="border:none;border-top:1px solid {BORDER};margin:0 0 1.2rem;">
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:{GOLD};margin-bottom:0.7rem;'>New Transaction</div>",
                    unsafe_allow_html=True)

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
            if insert_transaction(t_type, float(amount), category, note, tx_date):
                # Reload from DB so the new row gets its server-assigned ID
                st.session_state.df = load_data()
                st.success("Saved")
                st.rerun()

        st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:1.2rem 0;'>",
                    unsafe_allow_html=True)

        if not df.empty:
            csv_df = df.copy()
            csv_df["date"] = csv_df["date"].astype(str)
            st.download_button(
                "Export CSV",
                csv_df.to_csv(index=False),
                file_name="ledger_transactions.csv",
                mime="text/csv",
            )

        # Stats + sign-out
        if not df.empty:
            st.markdown(f"""
            <hr style='border:none;border-top:1px solid {BORDER};margin:1.2rem 0;'>
            <div style='font-size:0.7rem;color:{MUTED};line-height:1.9;margin-bottom:0.8rem;'>
              <div>{len(df)} transactions</div>
              <div>Since {df["date"].min().strftime("%b %Y")}</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Sign Out"):
            sign_out()

    return df


# ════════════════════════════════════════════════════════════════════════════
#  DASHBOARD TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_dashboard(df):
    df = ensure_datetime(df)
    now   = pd.Timestamp.now()
    cur_m = now.to_period("M").strftime("%Y-%m")
    prev_m= (now - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")
    def _m(m): return df[df["date"].dt.to_period("M").astype(str)==m]
    cur, prev = _m(cur_m), _m(prev_m)

    c_inc = cur[cur["type"]=="income"]["amount"].sum()
    c_exp = cur[cur["type"]=="expense"]["amount"].sum()
    c_net = c_inc - c_exp
    c_sr  = (c_net/c_inc*100) if c_inc>0 else 0
    p_exp = prev[prev["type"]=="expense"]["amount"].sum()

    rule("This Month")
    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Income", fmt_inr(c_inc))
    with k2:
        if p_exp > 0:
            chg = (c_exp-p_exp)/p_exp*100
            kpi("Expenses", fmt_inr(c_exp),
                f"{'▲' if chg>0 else '▼'} {abs(chg):.1f}% vs last month",
                "down" if chg>0 else "up")
        else:
            kpi("Expenses", fmt_inr(c_exp))
    with k3: kpi("Net Savings", fmt_inr(c_net),
                 f"{'▲' if c_net>=0 else '▼'} Net",
                 "up" if c_net>=0 else "down")
    with k4: kpi("Savings Rate", f"{c_sr:.1f}%", "Target: 20%",
                 "up" if c_sr>=20 else ("flat" if c_sr>=10 else "down"))

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.9])

    with left:
        rule("Spending Breakdown")
        cat_df = category_spend(df, month=cur_m)
        if not cat_df.empty:
            colours = [CAT_COLOURS.get(c, MUTED) for c in cat_df["category"]]
            fig = go.Figure(go.Pie(
                labels=cat_df["category"], values=cat_df["amount"],
                hole=0.60, marker=dict(colors=colours, line=dict(color=BLACK,width=2)),
                textinfo="percent", textfont=dict(size=10,color=WHITE),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                **base_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                              showlegend=True,
                              legend=dict(orientation="v",x=1.02,y=0.5,
                                          font=dict(size=10,color=MUTED))),
                annotations=[dict(text=f"<b>{fmt_inr(c_exp)}</b>",
                                  x=0.5,y=0.5,showarrow=False,
                                  font=dict(size=14,color=WHITE,
                                            family="Cormorant Garamond"))],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown(f"<div style='color:{MUTED};font-size:0.83rem;padding:2rem 0;'>No expenses this month.</div>",
                        unsafe_allow_html=True)

    with right:
        rule("Income vs Expenses")
        ms = monthly_summary(df)
        if not ms.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Income",x=ms["month"],y=ms["income"],
                                 marker_color=GREEN,opacity=0.75))
            fig.add_trace(go.Bar(name="Expenses",x=ms["month"],y=ms["expense"],
                                 marker_color=RED,opacity=0.75))
            fig.add_trace(go.Scatter(name="Net",x=ms["month"],y=ms["net"],
                                     mode="lines+markers",
                                     line=dict(color=GOLD,width=2),
                                     marker=dict(size=6,color=GOLD)))
            fig.update_layout(
                **base_layout(height=300, barmode="group"),
                xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
                yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    if not df.empty:
        rule("Recent Transactions")
        recent = df.sort_values("date",ascending=False).head(8).copy()
        recent["date"]   = recent["date"].dt.strftime("%d %b %Y")
        recent["amount"] = recent.apply(
            lambda r: f"+{fmt_inr(r['amount'])}" if r["type"]=="income"
                      else f"-{fmt_inr(r['amount'])}", axis=1)
        st.dataframe(
            recent[["date","category","note","amount"]].rename(columns={
                "date":"Date","category":"Category",
                "note":"Description","amount":"Amount"}),
            use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_analysis(df):
    df = ensure_datetime(df)
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No data yet. Add transactions from the sidebar.</div>",
                    unsafe_allow_html=True); return

    months = sorted(df["date"].dt.to_period("M").astype(str).unique(), reverse=True)
    sel    = st.selectbox("Period", ["All time"]+list(months), label_visibility="collapsed")
    cat_df = category_spend(df, month=(None if sel=="All time" else sel))

    rule("Category Breakdown")
    l, r = st.columns(2)
    with l:
        if not cat_df.empty:
            fig = px.bar(cat_df, x="amount", y="category", orientation="h",
                         color="category", color_discrete_map=CAT_COLOURS,
                         labels={"amount":"","category":""},
                         text=cat_df["amount"].apply(fmt_inr))
            fig.update_traces(textposition="outside", textfont=dict(size=10,color=MUTED))
            fig.update_layout(
                **base_layout(height=380, showlegend=False),
                yaxis=dict(categoryorder="total ascending",
                           gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
                xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with r:
        if not cat_df.empty:
            fig = px.treemap(cat_df, path=["category"], values="amount",
                             color="amount",
                             color_continuous_scale=[[0,CARD],[0.5,GOLD_DIM],[1,GOLD]])
            fig.update_layout(**base_layout(height=380, margin=dict(l=0,r=0,t=0,b=0)))
            fig.update_traces(textfont=dict(family="Inter",color=BLACK,size=11),
                              hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    rule("Month-over-Month Heatmap")
    pivot = df[df["type"]=="expense"].copy()
    pivot["month"] = pivot["date"].dt.to_period("M").astype(str)
    heat = pivot.pivot_table(index="category",columns="month",
                             values="amount",aggfunc="sum",fill_value=0)
    if not heat.empty:
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale=[[0,BLACK],[0.3,GOLD_DIM],[1,GOLD]],
            hovertemplate="<b>%{y}</b><br>%{x}<br>₹%{z:,.0f}<extra></extra>",showscale=True))
        fig.update_layout(
            **base_layout(height=340, margin=dict(l=140,r=20,t=20,b=20)),
            xaxis=dict(tickfont=dict(color=MUTED,size=10)),
            yaxis=dict(tickfont=dict(color=MUTED,size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    rule("Savings Rate by Month")
    ms = monthly_summary(df)
    if not ms.empty:
        colours = [GREEN if v>=20 else (GOLD if v>=0 else RED) for v in ms["savings_rate"]]
        fig = go.Figure(go.Bar(x=ms["month"],y=ms["savings_rate"],marker_color=colours,
                               text=[f"{v:.1f}%" for v in ms["savings_rate"]],
                               textposition="outside",textfont=dict(size=10,color=MUTED)))
        fig.add_hline(y=20,line_dash="dot",line_color=GOLD_DIM,
                      annotation_text="20% target",
                      annotation_font_color=GOLD_DIM,annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=260),
            yaxis=dict(title="",gridcolor=BORDER,linecolor=BORDER,
                       ticksuffix="%",tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
#  CASH FLOW TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_cashflow(df):
    df = ensure_datetime(df)
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No data yet.</div>",
                    unsafe_allow_html=True); return

    months    = sorted(df["date"].dt.to_period("M").astype(str).unique(), reverse=True)
    sel_month = st.selectbox("Month", months, label_visibility="collapsed")

    rule("Cash Flow Waterfall")
    month_df  = df[df["date"].dt.to_period("M").astype(str)==sel_month]
    inc_total = month_df[month_df["type"]=="income"]["amount"].sum()
    exp_cats  = month_df[month_df["type"]=="expense"] \
                  .groupby("category")["amount"].sum().sort_values(ascending=False)

    if not exp_cats.empty:
        net_val  = inc_total - exp_cats.sum()
        labels   = ["Income"] + list(exp_cats.index) + ["Net"]
        values   = [inc_total] + [-v for v in exp_cats.values] + [net_val]
        measures = ["absolute"] + ["relative"]*len(exp_cats) + ["total"]
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=measures, x=labels, y=values,
            connector=dict(line=dict(color=BORDER,width=1)),
            increasing=dict(marker_color=GREEN),
            decreasing=dict(marker_color=RED),
            totals=dict(marker_color=GOLD),
            textposition="outside",
            text=[fmt_inr(abs(v)) for v in values],
            textfont=dict(size=10,color=MUTED)))
        fig.update_layout(
            **base_layout(height=360, showlegend=False),
            yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    rule("Daily Spending Pattern")
    daily = daily_spend(df, sel_month)
    if not daily.empty:
        avg_daily = daily["amount"].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"],y=daily["amount"],
                             marker_color=GOLD_DIM,name="Daily spend"))
        fig.add_hline(y=avg_daily,line_dash="dot",line_color=GOLD,
                      annotation_text=f"Avg {fmt_inr(avg_daily)}/day",
                      annotation_font_color=GOLD,annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=240),
            yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    rule("Cumulative Spend vs Income")
    month_exp = df[(df["type"]=="expense") &
                   (df["date"].dt.to_period("M").astype(str)==sel_month)] \
                  .sort_values("date").copy()
    if not month_exp.empty:
        month_exp["cumulative"] = month_exp["amount"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=month_exp["date"],y=month_exp["cumulative"],
            mode="lines",name="Cumulative spend",
            line=dict(color=RED,width=2),
            fill="tozeroy",fillcolor="rgba(207,102,121,0.08)"))
        fig.add_hline(y=inc_total,line_dash="dot",line_color=GREEN,
                      annotation_text=f"Income {fmt_inr(inc_total)}",
                      annotation_font_color=GREEN,annotation_font_size=10)
        fig.update_layout(
            **base_layout(height=220),
            yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
            xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
#  PREDICTIONS TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_predictions(df):
    df = ensure_datetime(df)
    ms = monthly_summary(df)
    if len(ms) < 3:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>Need at least 3 months of data. You have {len(ms)} so far.</div>",
                    unsafe_allow_html=True); return

    preds = predict_expenses(df, n=3)
    rule("3-Month Expense Forecast")
    if not preds.empty:
        avg_income = ms["income"].mean()
        cols = st.columns(3)
        for i, (_, row) in enumerate(preds.iterrows()):
            diff = row["predicted"] - ms["expense"].mean()
            with cols[i]:
                kpi(row["month"], fmt_inr(row["predicted"]),
                    f"{'▲' if diff>0 else '▼'} {fmt_inr(abs(diff))} vs avg",
                    "down" if diff>0 else "up")

    rule("Trend & Forecast")
    fig = go.Figure()
    if not preds.empty:
        fig.add_trace(go.Scatter(
            x=list(preds["month"])+list(reversed(preds["month"])),
            y=list(preds["upper"])+list(reversed(preds["lower"])),
            fill="toself", fillcolor="rgba(201,168,76,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ms["month"],y=ms["expense"],
        mode="lines+markers",name="Actual expenses",
        line=dict(color=RED,width=2),marker=dict(size=6,color=RED)))
    fig.add_trace(go.Scatter(x=ms["month"],y=ms["income"],
        mode="lines+markers",name="Income",
        line=dict(color=GREEN,width=2),marker=dict(size=6,color=GREEN)))
    if not preds.empty:
        bx = [ms["month"].iloc[-1]]+list(preds["month"])
        by = [ms["expense"].iloc[-1]]+list(preds["predicted"])
        fig.add_trace(go.Scatter(x=bx,y=by,mode="lines+markers",name="Forecast",
            line=dict(color=GOLD,width=2,dash="dot"),
            marker=dict(size=7,color=GOLD,symbol="diamond")))
    fig.update_layout(
        **base_layout(height=360),
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
        xaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=MUTED)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    if not preds.empty:
        avg_income = ms["income"].mean()
        rule("Projected Savings (at avg income)")
        s_cols = st.columns(3)
        for i, (_, row) in enumerate(preds.iterrows()):
            proj_net = avg_income - row["predicted"]
            proj_sr  = (proj_net/avg_income*100) if avg_income>0 else 0
            with s_cols[i]:
                kpi(f"{row['month']} savings", fmt_inr(proj_net),
                    f"{proj_sr:.1f}% savings rate",
                    "up" if proj_sr>=20 else "down")

    st.markdown(f"""
    <div style='font-size:0.75rem;color:{MUTED};margin-top:1.5rem;padding:0.85rem 1rem;
    background:{CARD};border-left:2px solid {BORDER};'>
    Forecast uses linear regression on monthly expense totals. Treat projections as
    directional indicators. Accuracy improves with more months of data.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHTS TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_insights(df):
    df = ensure_datetime(df)
    rule("Financial Insights")
    for ins in generate_insights(df):
        insight_card(ins["kind"], ins["title"], ins["body"])

    if df.empty: return
    st.markdown("<br>", unsafe_allow_html=True)
    rule("Financial Principles")
    ms = monthly_summary(df)
    avg_exp = ms["expense"].mean() if not ms.empty else 0
    tips = [
        ("50 / 30 / 20 rule",
         "50% to needs (rent, groceries, utilities), 30% to wants, 20% to savings. Revisit every quarter."),
        ("Pay yourself first",
         "Move your savings target to a separate account on salary day — before any discretionary spending."),
        ("Emergency fund",
         f"Your 4-month buffer target is approximately {fmt_inr(avg_exp*4)}. Keep it in a liquid instrument."),
        ("Index fund SIP",
         "₹5,000/month in an index fund at 12% CAGR grows to ~₹50 lakh over 20 years. Start early."),
        ("Subscription audit",
         "List every recurring charge quarterly. Cancel any unused in the past 30 days — they add up."),
        ("Lifestyle inflation",
         "When income rises, resist scaling expenses proportionally. Route salary hikes into investments."),
    ]
    cols = st.columns(2)
    for i, (title, body) in enumerate(tips):
        with cols[i%2]: tip_card(title, body)


# ════════════════════════════════════════════════════════════════════════════
#  TRANSACTIONS TAB
# ════════════════════════════════════════════════════════════════════════════

def tab_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime(df)
    if df.empty:
        st.markdown(f"<div style='color:{MUTED};padding:2rem 0;'>No transactions yet.</div>",
                    unsafe_allow_html=True); return df

    rule("Filter")
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1: t_filter = st.selectbox("Type",["All","income","expense"])
    with fc2:
        months   = ["All"]+sorted(df["date"].dt.to_period("M").astype(str).unique(),reverse=True)
        m_filter = st.selectbox("Month", months)
    with fc3:
        cats     = ["All"]+sorted(df["category"].unique())
        c_filter = st.selectbox("Category", cats)
    with fc4:
        search   = st.text_input("Search description", placeholder="keyword…")

    filtered = df.copy()
    if t_filter != "All": filtered = filtered[filtered["type"]==t_filter]
    if m_filter != "All": filtered = filtered[filtered["date"].dt.to_period("M").astype(str)==m_filter]
    if c_filter != "All": filtered = filtered[filtered["category"]==c_filter]
    if search:            filtered = filtered[filtered["note"].str.contains(search,case=False,na=False)]

    rule("Summary")
    s1,s2,s3 = st.columns(3)
    with s1: kpi("Transactions", str(len(filtered)))
    with s2: kpi("Total Income",   fmt_inr(filtered[filtered["type"]=="income"]["amount"].sum()))
    with s3: kpi("Total Expenses", fmt_inr(filtered[filtered["type"]=="expense"]["amount"].sum()))

    rule("Ledger")
    display = filtered.sort_values("date",ascending=False).copy()
    display["date"]   = display["date"].dt.strftime("%d %b %Y")
    display["amount"] = display.apply(
        lambda r: f"+{fmt_inr(r['amount'])}" if r["type"]=="income"
                  else f"-{fmt_inr(r['amount'])}", axis=1)
    st.dataframe(
        display[["id","date","type","category","note","amount"]].rename(columns={
            "id":"#","date":"Date","type":"Type",
            "category":"Category","note":"Description","amount":"Amount"}),
        use_container_width=True, hide_index=True, height=420)
    st.caption(f"{len(filtered)} of {len(df)} transactions")

    rule("Delete Transaction")
    d1,d2 = st.columns([1,3])
    with d1:
        del_id = st.number_input("Transaction #", min_value=1, step=1)
    with d2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Delete"):
            if delete_transaction_db(int(del_id)):
                st.session_state.df = load_data()
                st.success(f"Transaction #{del_id} deleted.")
                st.rerun()
            else:
                st.error("Not found or not yours.")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Auth gate ─────────────────────────────────────────
    if "user" not in st.session_state:
        render_auth_page()   # calls st.stop() if not authenticated

    # ── Load data once per session (or after mutations) ───
    if "df" not in st.session_state:
        st.session_state.df = load_data()

    render_sidebar(st.session_state.df)
    df = st.session_state.df

    # ── Page header ───────────────────────────────────────
    st.markdown(f"""
    <div class="page-header">
      <span class="page-wordmark">Ledger</span>
      <span class="page-sub">Personal Finance Intelligence</span>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Dashboard","Analysis","Cash Flow","Predictions","Insights","Transactions"])
    with tabs[0]: tab_dashboard(df)
    with tabs[1]: tab_analysis(df)
    with tabs[2]: tab_cashflow(df)
    with tabs[3]: tab_predictions(df)
    with tabs[4]: tab_insights(df)
    with tabs[5]: tab_transactions(df)


if __name__ == "__main__":
    main()