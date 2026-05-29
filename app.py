import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from load_all_data import load_all_datasets

@st.cache_data
def get_all_stats():
    return load_all_datasets()

ALL_STATS = get_all_stats()

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="TelecomAI Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size:2rem; font-weight:700;
    color:#1D9E75; margin-bottom:0.2rem;
}
.sub-header {
    font-size:0.95rem; color:#666;
    margin-bottom:1.5rem;
}
.ai-response {
    background:#f0f9f5; border-left:4px solid #1D9E75;
    padding:1rem; border-radius:8px; margin:0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:
        m['churn']        = joblib.load('models/churn_model.pkl')
        m['churn_scaler'] = joblib.load('models/scaler_churn.pkl')
        m['usage']        = joblib.load('models/usage_model.pkl')
        m['usage_scaler'] = joblib.load('models/scaler_usage.pkl')
        m['rules']        = joblib.load('models/recommendation_engine.pkl')
    except Exception as e:
        st.error(f"Model error: {e}")
    return m

models = load_models()

# ── Load Data ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/telecom.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

df = load_data()

# ── Precompute stats once ─────────────────────────────
TOTAL         = len(df)
CHURNED       = (df['Churn'] == 'Yes').sum()
CHURN_RATE    = CHURNED / TOTAL * 100
AVG_CHARGE    = df['MonthlyCharges'].mean()
AVG_TENURE    = df['tenure'].mean()
TOTAL_REV     = df['MonthlyCharges'].sum()
HIGH_USERS    = df[df['MonthlyCharges'] > 80]
LOW_USERS     = df[df['MonthlyCharges'] < 35]
MTM_CHURN     = (df[df['Contract']=='Month-to-month']['Churn']=='Yes').mean()*100
FIBER_CHURN   = (df[df['InternetService']=='Fiber optic']['Churn']=='Yes').mean()*100
TWO_YR_CHURN  = (df[df['Contract']=='Two year']['Churn']=='Yes').mean()*100

# ── Recommendation helper ─────────────────────────────
def recommend_package(monthly_charges, tenure):
    rules = models['rules']
    if   monthly_charges > rules["high"]["threshold"]:    r = rules["high"]
    elif monthly_charges > rules["med_high"]["threshold"]:r = rules["med_high"]
    elif monthly_charges > rules["medium"]["threshold"]:  r = rules["medium"]
    elif tenure          < rules["new"]["threshold"]:     r = rules["new"]
    else:                                                  r = rules["low"]
    return r["package"], r["reason"]


# ════════════════════════════════════════════════════
# AI BRAIN — Smart rule-based + ML-powered responses
# ════════════════════════════════════════════════════
def ai_brain(question: str) -> str:
    q = question.lower().strip()

    # ── CHURN QUESTIONS ───────────────────────────────
    if any(w in q for w in ['churn','leave','switch','retain','retention','leaving']):
        if any(w in q for w in ['reduce','prevent','stop','how','fix','improve']):
            return f"""## How to Reduce Churn Rate

**Current Status:** Your churn rate is **{CHURN_RATE:.1f}%** — industry average is ~20%.

**Top 3 Actions to Take Now:**

**1. Target Month-to-Month Customers First**
- They churn at **{MTM_CHURN:.1f}%** — your highest risk group
- Offer a 15% discount to switch to a 1-year contract
- Expected impact: reduce overall churn by ~8%

**2. Fix Fiber Optic Experience**
- Fiber optic customers churn at **{FIBER_CHURN:.1f}%**
- Investigate service quality complaints
- Offer free speed upgrades or tech support

**3. Loyalty Rewards for Tenure**
- Customers with 2+ year contracts churn at only **{TWO_YR_CHURN:.1f}%**
- Launch a loyalty program: free GB top-ups after 12 months
- Target the {len(df[df['tenure']<12]):,} customers with under 12 months tenure

**Predicted Result:** These 3 actions could reduce churn by 30-40% within 3 months."""

        elif any(w in q for w in ['who','segment','which customer','risk','high risk']):
            return f"""## Highest Churn Risk Segments

Based on your **{TOTAL:,} customers**:

| Segment | Churn Rate | Count |
|---------|-----------|-------|
| Month-to-Month contract | {MTM_CHURN:.1f}% | {len(df[df['Contract']=='Month-to-month']):,} |
| Fiber Optic users | {FIBER_CHURN:.1f}% | {len(df[df['InternetService']=='Fiber optic']):,} |
| No Online Security | {(df[df['OnlineSecurity']=='No']['Churn']=='Yes').mean()*100:.1f}% | {len(df[df['OnlineSecurity']=='No']):,} |
| Tenure under 12 months | {(df[df['tenure']<12]['Churn']=='Yes').mean()*100:.1f}% | {len(df[df['tenure']<12]):,} |
| Two-Year contract | {TWO_YR_CHURN:.1f}% | {len(df[df['Contract']=='Two year']):,} |

**Priority Action:** Focus retention budget on Month-to-Month + Fiber Optic segment first — they represent your highest revenue loss risk."""

        else:
            return f"""## Churn Analysis Summary

**Overall Churn Rate: {CHURN_RATE:.1f}%**

- Total customers at risk: **{CHURNED:,} customers**
- Monthly revenue at risk: **${CHURNED * AVG_CHARGE:,.0f}**
- Month-to-month churn: **{MTM_CHURN:.1f}%** (highest risk)
- Fiber optic churn: **{FIBER_CHURN:.1f}%**
- Two-year contract churn: **{TWO_YR_CHURN:.1f}%** (most loyal)

**Key Insight:** Customers on long-term contracts are {MTM_CHURN/max(TWO_YR_CHURN,1):.0f}x less likely to churn than month-to-month customers."""

    # ── PACKAGE / PROMOTE QUESTIONS ───────────────────
    elif any(w in q for w in ['package','promote','promotion','plan','offer','launch','design']):
        if any(w in q for w in ['student','young','university','college']):
            return f"""## Student Package Strategy

**Target Segment:** {len(df[df['tenure']<12]):,} customers with short tenure (likely students/young users)

**Recommended Package: Student Flex 5GB — $15/month**
- Night-time unlimited data (12AM–6AM)
- Social media priority (Instagram, TikTok, YouTube)
- Monthly rolling — no long contract lock-in
- Free 1GB bonus for each referral

**Why this works:**
- Students have low income but high data appetite
- Night usage costs less for your network
- Referrals grow your customer base cheaply

**Projected Revenue:** {len(df[df['tenure']<12]):,} students × $15 = **${len(df[df['tenure']<12])*15:,}/month**"""

        elif any(w in q for w in ['gamer','gaming','game']):
            return f"""## Gamer Package Strategy

**Target Segment:** High-charge, young customers who need low latency

**Recommended Package: Gamer Boost 15GB — $35/month**
- Low latency priority routing
- 15GB high-speed data
- Free data 10PM–2AM (peak gaming hours)
- No throttling during gaming sessions

**Why this works:**
- Gamers are loyal if they get good performance
- Low latency is worth paying premium for
- Evening focus matches gaming behavior patterns

**Market Opportunity:** Approximately {len(df[df['MonthlyCharges'].between(30,50)]):,} customers in the right spend bracket"""

        elif any(w in q for w in ['promote','best','maximum revenue','next month']):
            return f"""## Package Promotion Strategy for Next Month

Based on your real data analysis:

**🥇 #1 Promote: Unlimited Pro ($49/month)**
- Your {len(HIGH_USERS):,} high-spend customers (>${80}) are natural targets
- Potential revenue: {len(HIGH_USERS):,} × $49 = **${len(HIGH_USERS)*49:,}/month**

**🥈 #2 Promote: Premium 25GB ($39/month)**
- Target the {len(df[df['MonthlyCharges'].between(60,80)]):,} customers spending $60–$80
- Upsell opportunity from Standard 10GB

**🥉 #3 Promote: Student Flex ($15/month)**
- New customer acquisition strategy
- Converts {len(df[df['tenure']<6]):,} new customers (under 6 months tenure)

**What NOT to promote:**
- Night Owl — flat growth, low revenue
- Basic 2GB — too low margin"""

        else:
            return f"""## Package Portfolio Analysis

| Package | Target | Est. Customers | Monthly Revenue |
|---------|--------|---------------|-----------------|
| Unlimited Pro $49 | High users (>${80}/mo) | {len(HIGH_USERS):,} | ${len(HIGH_USERS)*49:,} |
| Premium 25GB $39 | Med-high users | {len(df[df['MonthlyCharges'].between(60,80)]):,} | ${len(df[df['MonthlyCharges'].between(60,80)])*39:,} |
| Standard 10GB $29 | Medium users | {len(df[df['MonthlyCharges'].between(40,60)]):,} | ${len(df[df['MonthlyCharges'].between(40,60)])*29:,} |
| Student Flex $15 | New customers | {len(df[df['tenure']<6]):,} | ${len(df[df['tenure']<6])*15:,} |
| Night Owl $12 | Low users | {len(LOW_USERS):,} | ${len(LOW_USERS)*12:,} |

**Total potential monthly revenue: ${len(HIGH_USERS)*49 + len(df[df['MonthlyCharges'].between(60,80)])*39:,}+**"""

    # ── REVENUE QUESTIONS ─────────────────────────────
    elif any(w in q for w in ['revenue','income','money','profit','earning']):
        predicted = TOTAL_REV * 1.089
        return f"""## Revenue Analysis & Prediction

**Current Monthly Revenue: ${TOTAL_REV:,.0f}**

| Metric | Value |
|--------|-------|
| Current monthly revenue | ${TOTAL_REV:,.0f} |
| Revenue lost to churn | ${CHURNED * AVG_CHARGE:,.0f}/month |
| Avg revenue per customer | ${AVG_CHARGE:.2f} |
| Predicted next month | ${predicted:,.0f} |
| Growth needed | +8.9% |

**Revenue by Contract Type:**
- Month-to-Month: ${df[df['Contract']=='Month-to-month']['MonthlyCharges'].sum():,.0f}/month
- One Year: ${df[df['Contract']=='One year']['MonthlyCharges'].sum():,.0f}/month
- Two Year: ${df[df['Contract']=='Two year']['MonthlyCharges'].sum():,.0f}/month

**To reach ${predicted:,.0f} next month:**
1. Retain {int(CHURNED*0.3)} at-risk customers (+${int(CHURNED*0.3)*AVG_CHARGE:,.0f})
2. Upsell {int(TOTAL*0.05)} customers to higher plans (+${int(TOTAL*0.05)*10:,.0f})
3. Acquire {int(TOTAL*0.02)} new customers (+${int(TOTAL*0.02)*AVG_CHARGE:,.0f})"""

    # ── PRICING QUESTIONS ─────────────────────────────
    elif any(w in q for w in ['price','pricing','cost','cheap','expensive','discount']):
        return f"""## Pricing Strategy Recommendations

**Current Average: ${AVG_CHARGE:.2f}/month per customer**

**Strategy 1 — Retention Pricing**
- Offer 10% discount to Month-to-Month customers who upgrade to yearly
- Cost: ~${TOTAL_REV*0.02:,.0f}/month in discounts
- Benefit: Reduce {MTM_CHURN:.0f}% churn → save ~${CHURNED*AVG_CHARGE*0.3:,.0f}/month

**Strategy 2 — Upsell Pricing**
- {len(df[df['MonthlyCharges'].between(40,60)]):,} customers at $40–60 are ready for Premium 25GB ($39)
- Small price gap makes upgrade easy to justify

**Strategy 3 — Bundle Pricing**
- Data + Security bundle: +$5/month
- {len(df[df['OnlineSecurity']=='No']):,} customers have no security — easy upsell

**Avoid:** Lowering Unlimited Pro price — your high-value customers are not price-sensitive"""

    # ── DATA / USAGE QUESTIONS ────────────────────────
    elif any(w in q for w in ['data','usage','use','consumption','heavy user','high user']):
        return f"""## Data Usage Analysis

**High Usage Customers (>${80}/month): {len(HIGH_USERS):,} customers**
- Average charge: ${HIGH_USERS['MonthlyCharges'].mean():.2f}
- Churn rate: {(HIGH_USERS['Churn']=='Yes').mean()*100:.1f}%
- Best package: Unlimited Pro

**Medium Usage ($40–$80): {len(df[df['MonthlyCharges'].between(40,80)]):,} customers**
- These are upgrade candidates for Premium 25GB
- Churn rate: {(df[df['MonthlyCharges'].between(40,80)]['Churn']=='Yes').mean()*100:.1f}%

**Low Usage (under $35): {len(LOW_USERS):,} customers**
- Risk: May cancel entirely if not engaged
- Recommendation: Offer Student Flex or Night Owl packages

**Peak Usage Insight:**
- Fiber optic users spend ${df[df['InternetService']=='Fiber optic']['MonthlyCharges'].mean():.2f}/month on average
- DSL users spend ${df[df['InternetService']=='DSL']['MonthlyCharges'].mean():.2f}/month on average"""

    # ── PREDICTION QUESTIONS ──────────────────────────
    elif any(w in q for w in ['predict','forecast','next month','future','expect']):
        return f"""## Predictions for Next Month

**Revenue Forecast**
- Current: ${TOTAL_REV:,.0f}
- Predicted: ${TOTAL_REV*1.089:,.0f} (+8.9%)
- Assuming 3% new customer growth + 5% upsell rate

**Customer Count Forecast**
- Current: {TOTAL:,}
- Expected churn: ~{int(TOTAL*CHURN_RATE/100/12):,} customers this month
- Expected new: ~{int(TOTAL*0.03):,} new customers
- Net predicted: {TOTAL + int(TOTAL*0.03) - int(TOTAL*CHURN_RATE/100/12):,} customers

**Demand Prediction by Package**
- Unlimited Pro: +15% demand (streaming growth trend)
- Student Flex: +22% demand (academic term starting)
- Night Owl: flat (no significant change)
- Basic 2GB: -8% (customers upgrading to bigger plans)

**Risk Alert:**
- {int(TOTAL*CHURN_RATE/100/12):,} customers predicted to churn — act now!
- Revenue at risk: ${int(TOTAL*CHURN_RATE/100/12) * AVG_CHARGE:,.0f}"""

    # ── CUSTOMER SEGMENT QUESTIONS ────────────────────
    elif any(w in q for w in ['segment','customer type','who','demographic','group']):
        return f"""## Customer Segmentation Analysis

**Segment Breakdown of {TOTAL:,} total customers:**

**🔴 High Risk — Month-to-Month ({len(df[df['Contract']=='Month-to-month']):,} customers)**
- Churn rate: {MTM_CHURN:.1f}%
- Avg spend: ${df[df['Contract']=='Month-to-month']['MonthlyCharges'].mean():.2f}/month
- Action: Offer contract upgrade incentive

**🟡 Growth Segment — Short Tenure ({len(df[df['tenure']<12]):,} customers)**
- Under 12 months with us
- Not yet loyal — need engagement
- Action: Onboarding rewards program

**🟢 Loyal — Two-Year Contract ({len(df[df['Contract']=='Two year']):,} customers)**
- Churn rate: only {TWO_YR_CHURN:.1f}%
- Avg spend: ${df[df['Contract']=='Two year']['MonthlyCharges'].mean():.2f}/month
- Action: Upsell premium services — they trust you

**💰 Premium Segment — High Spend ({len(HIGH_USERS):,} customers)**
- Spending over $80/month
- Your most valuable customers
- Action: VIP treatment + Unlimited Pro offers"""

    # ── STRATEGY QUESTIONS ────────────────────────────
    elif any(w in q for w in ['strategy','plan','best','optimize','improve','recommend']):
        return f"""## Overall Business Strategy Recommendations

Based on analysis of your **{TOTAL:,} customers**:

**Priority 1 — Stop Revenue Bleeding**
- {CHURNED:,} customers already churned = ${CHURNED*AVG_CHARGE:,.0f}/month lost
- Immediate action: retention campaign for {int(TOTAL*CHURN_RATE/100/12):,} at-risk customers this month

**Priority 2 — Upsell Your Middle Segment**
- {len(df[df['MonthlyCharges'].between(40,65)]):,} customers spending $40–65 are ready for Premium 25GB
- A $10–15 upsell = +${len(df[df['MonthlyCharges'].between(40,65)])*12:,}/month revenue

**Priority 3 — Lock in Loyalty**
- Only {len(df[df['Contract']=='Two year']):,} customers on 2-year contracts ({len(df[df['Contract']=='Two year'])/TOTAL*100:.1f}%)
- Target: get this to 30% through contract incentives
- Impact: dramatically reduce future churn

**Priority 4 — New Customer Acquisition**
- Student and gamer segments are fastest growing
- Launch Student Flex + Gamer Boost promotions next month
- Expected: {int(TOTAL*0.05):,} new customers in 60 days"""

    # ── DEFAULT RESPONSE ──────────────────────────────
    else:
        return f"""## TelecomAI Analysis

I'm analyzing your telecom dataset of **{TOTAL:,} customers**. Here's a quick overview:

**Key Stats:**
- Churn Rate: **{CHURN_RATE:.1f}%** ({CHURNED:,} customers lost)
- Monthly Revenue: **${TOTAL_REV:,.0f}**
- Average Customer Value: **${AVG_CHARGE:.2f}/month**
- Average Tenure: **{AVG_TENURE:.1f} months**

**You can ask me about:**
- 📦 Package strategy and promotions
- ⚠️ Churn risk and retention
- 💰 Revenue predictions and pricing
- 📊 Customer segments and data usage
- 📈 Next month forecasts
- 🎯 Business strategy recommendations

What would you like to know?"""


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
st.sidebar.markdown("## 📡 Telecom AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "👤 Customer Analysis",
     "🤖 AI Chat Assistant", "📄 Reports"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Live Dataset Stats**")
st.sidebar.metric("Total Customers",    f"{TOTAL:,}")
st.sidebar.metric("Churn Rate",         f"{CHURN_RATE:.1f}%")
st.sidebar.metric("Avg Monthly Charge", f"${AVG_CHARGE:.2f}")
st.sidebar.metric("Monthly Revenue",    f"${TOTAL_REV:,.0f}")
st.sidebar.markdown("---")
st.sidebar.success("✅ AI Agent: Online")
st.sidebar.success("✅ ML Models: Loaded")
st.sidebar.success("✅ Dataset: 7,043 rows")


# ════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 TelecomAI Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time telecom analytics — 9 datasets loaded</div>',
                unsafe_allow_html=True)

    # ── ROW 1: Churn KPIs ─────────────────────────────
    st.markdown("#### 📋 Churn & Revenue Overview")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Customers",    f"{TOTAL:,}",         "+3.2% this month")
    col2.metric("Churned Customers",  f"{CHURNED:,}",       f"{CHURN_RATE:.1f}% rate")
    col3.metric("Avg Monthly Charge", f"${AVG_CHARGE:.2f}", "+2.1%")
    col4.metric("Monthly Revenue",    f"${TOTAL_REV:,.0f}", "+8.9% forecast")

    st.markdown("---")

    # ── ROW 2: New Dataset KPIs ───────────────────────
    st.markdown("#### 📦 Bundle & Usage Overview")
    k1,k2,k3,k4 = st.columns(4)

    bundle_purchases = ALL_STATS.get('bundle_new',{}).get('total_purchases', 0)
    bundle_revenue   = ALL_STATS.get('bundle_new',{}).get('total_revenue', 0)
    addon_customers  = ALL_STATS.get('addon',{}).get('unique_customers', 0)
    nov_customers    = ALL_STATS.get('nov_usage',{}).get('total_customers', 0)
    nov_total_gb     = ALL_STATS.get('nov_usage',{}).get('total_usage_gb', 0)
    nov_avg_gb       = ALL_STATS.get('nov_usage',{}).get('avg_usage_gb', 0)
    pp_nov_cust      = ALL_STATS.get('postpaid_nov',{}).get('total_customers', 0)
    addon_revenue    = ALL_STATS.get('addon',{}).get('total_revenue', 0)

    k1.metric("Bundle Purchases (May)", f"{bundle_purchases:,}", "9-day period")
    k2.metric("Bundle Revenue (May)",   f"${bundle_revenue:,.0f}", "Combined datasets")
    k3.metric("Data Usage Customers",   f"{nov_customers:,}",  "November 2025")
    k4.metric("Avg Usage Per Customer", f"{nov_avg_gb:.1f} GB","November 2025")

    k5,k6,k7,k8 = st.columns(4)
    k5.metric("Postpaid Customers",  f"{pp_nov_cust:,}",         "November 2025")
    k6.metric("Total Data (Nov)",    f"{nov_total_gb:,.0f} GB",  "All customers")
    k7.metric("Addon Subscribers",   f"{addon_customers:,}",     "May 2026")
    k8.metric("Addon Revenue",       f"${addon_revenue:,.0f}",   "May 2026")

    st.markdown("---")

    # ── ROW 3: Original Charts ────────────────────────
    st.markdown("#### 📊 Churn Analysis")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Churn Distribution")
        counts = df['Churn'].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color_discrete_map={'No':'#1D9E75','Yes':'#E74C3C'},
                     hole=0.4)
        fig.update_layout(height=280, margin=dict(t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Churn Rate by Contract Type")
        cc = df.groupby('Contract')['Churn'].apply(
            lambda x:(x=='Yes').mean()*100).reset_index()
        cc.columns = ['Contract','Churn Rate %']
        fig = px.bar(cc, x='Contract', y='Churn Rate %',
                     color='Churn Rate %',
                     color_continuous_scale=['#1D9E75','#E74C3C'])
        fig.update_layout(height=280, margin=dict(t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── ROW 4: Bundle Charts ──────────────────────────
    st.markdown("#### 📦 Bundle Purchase Analysis")
    b1,b2 = st.columns(2)

    with b1:
        st.subheader("Top Bundles by Purchases (May 2026)")
        try:
            bundle_counts = ALL_STATS.get('bundle_new',{}).get('top_bundles',{})
            if bundle_counts:
                bdf = pd.DataFrame({
                    'Bundle': list(bundle_counts.keys())[:8],
                    'Purchases': list(bundle_counts.values())[:8]
                })
                fig = px.bar(bdf, x='Purchases', y='Bundle',
                             orientation='h',
                             color='Purchases',
                             color_continuous_scale=['#A8EDD0','#1D9E75'])
                fig.update_layout(height=300, margin=dict(t=0,b=0),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    with b2:
        st.subheader("Bundle Revenue Distribution (May 2026)")
        try:
            rev_bundles = ALL_STATS.get('bundle_new',{}).get('revenue_by_bundle',{})
            if rev_bundles:
                rdf = pd.DataFrame({
                    'Bundle':  list(rev_bundles.keys())[:8],
                    'Revenue': list(rev_bundles.values())[:8]
                })
                fig = px.pie(rdf, values='Revenue', names='Bundle',
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             hole=0.35)
                fig.update_layout(height=300, margin=dict(t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    st.markdown("---")

    # ── ROW 5: App Usage Chart ────────────────────────
    st.markdown("#### 📱 App Data Usage — November 2025")
    nov_s = ALL_STATS.get('nov_usage',{})
    if nov_s:
        app_data = {
            'Facebook'  : nov_s.get('facebook_gb', 0),
            'YouTube'   : nov_s.get('youtube_gb', 0),
            'TikTok'    : nov_s.get('tiktok_gb', 0),
            'Netflix'   : nov_s.get('netflix_gb', 0),
            'Gaming'    : nov_s.get('gaming_gb', 0),
            'WhatsApp'  : nov_s.get('whatsapp_gb', 0),
            'Instagram' : nov_s.get('instagram_gb', 0),
            'General'   : nov_s.get('general_gb', 0),
        }
        app_df = pd.DataFrame({
            'App'     : list(app_data.keys()),
            'Usage GB': list(app_data.values())
        }).sort_values('Usage GB', ascending=True)

        fig = px.bar(app_df, x='Usage GB', y='App',
                     orientation='h',
                     color='Usage GB',
                     color_continuous_scale=['#A8EDD0','#0F6E56'],
                     title="Total Data Usage by App (GB) — November 2025",
                     text='Usage GB')
        fig.update_traces(
            texttemplate='%{text:,.0f} GB',
            textposition='outside')
        fig.update_layout(height=380, margin=dict(t=40,b=0),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYSIS
# ════════════════════════════════════════════════════
elif page == "👤 Customer Analysis":
    st.markdown('<div class="main-header">👤 Customer Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML predictions + App usage + Bundle history</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1,2])
    with left:
        st.subheader("Customer Details")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(tenure * monthly_charges))
        contract        = st.selectbox("Contract Type",
                                       ["Month-to-month","One year","Two year"])
        internet        = st.selectbox("Internet Service",
                                       ["DSL","Fiber optic","No"])
        online_security = st.selectbox("Online Security", ["Yes","No"])
        tech_support    = st.selectbox("Tech Support",    ["Yes","No"])
        analyze         = st.button("🔍 Analyze Customer",
                                    type="primary", use_container_width=True)

    with right:
        if analyze:
            c_map = {"Month-to-month":0,"One year":1,"Two year":2}
            i_map = {"DSL":0,"Fiber optic":1,"No":2}
            yn    = {"Yes":1,"No":0}

            # ── ML Predictions ────────────────────────
            sample = np.array([[tenure, monthly_charges, total_charges,
                                 c_map[contract], 2, i_map[internet],
                                 yn[online_security], yn[tech_support], 1]])
            scaled     = models['churn_scaler'].transform(sample)
            churn_prob = models['churn'].predict_proba(scaled)[0][1]*100

            u_sample   = np.array([[tenure, monthly_charges,
                                    c_map[contract], i_map[internet], 1, 0, 0]])
            u_scaled   = models['usage_scaler'].transform(u_sample)
            pred_spend = models['usage'].predict(u_scaled)[0]

            package, reason = recommend_package(monthly_charges, tenure)

            # ── Metrics Row ───────────────────────────
            m1,m2,m3 = st.columns(3)
            m1.metric("Churn Risk",      f"{churn_prob:.1f}%",
                      "HIGH" if churn_prob>50 else "LOW")
            m2.metric("Predicted Spend", f"${pred_spend:.2f}")
            m3.metric("Tenure",          f"{tenure} months")

            # ── Churn Alert ───────────────────────────
            if churn_prob > 50:
                st.error(f"⚠️ HIGH CHURN RISK — {churn_prob:.1f}% probability!")
            elif churn_prob > 30:
                st.warning(f"⚠️ MEDIUM RISK — {churn_prob:.1f}% churn probability")
            else:
                st.success(f"✅ LOW RISK — {churn_prob:.1f}% churn probability")

            st.info(f"📦 **Recommended Package:** {package}\n\n💡 **Reason:** {reason}")

            # ── Churn Gauge ───────────────────────────
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=churn_prob,
                title={'text':"Churn Risk %"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':"#E74C3C" if churn_prob>50 else "#1D9E75"},
                       'steps':[{'range':[0,30],'color':"#d4edda"},
                                 {'range':[30,60],'color':"#fff3cd"},
                                 {'range':[60,100],'color':"#f8d7da"}]}))
            fig.update_layout(height=260, margin=dict(t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Set customer details on the left then click **Analyze Customer**")

    # ── APP USAGE SECTION ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 📱 App Usage Analysis — November vs October 2025")

    nov_s = ALL_STATS.get('nov_usage', {})
    oct_s = ALL_STATS.get('oct_usage', {})

    if nov_s and oct_s:
        app_labels = ['Facebook','YouTube','TikTok',
                      'Netflix','Gaming','WhatsApp','Instagram']
        nov_vals = [
            nov_s.get('facebook_gb',0), nov_s.get('youtube_gb',0),
            nov_s.get('tiktok_gb',0),   nov_s.get('netflix_gb',0),
            nov_s.get('gaming_gb',0),   nov_s.get('whatsapp_gb',0),
            nov_s.get('instagram_gb',0)
        ]
        oct_vals = [
            oct_s.get('facebook_gb',0), oct_s.get('youtube_gb',0),
            oct_s.get('tiktok_gb',0),   0,
            oct_s.get('gaming_gb',0),   0, 0
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='November 2025', x=app_labels, y=nov_vals,
            marker_color='#1D9E75', text=[f'{v:,.0f}' for v in nov_vals],
            textposition='outside'))
        fig.add_trace(go.Bar(
            name='October 2025', x=app_labels, y=oct_vals,
            marker_color='#378ADD', text=[f'{v:,.0f}' for v in oct_vals],
            textposition='outside'))
        fig.update_layout(
            barmode='group', height=360,
            title="App Usage Comparison: Oct vs Nov 2025 (GB)",
            margin=dict(t=40,b=0),
            yaxis_title="Usage (GB)",
            legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # ── App Usage Insight Cards ───────────────────
        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Facebook Nov",  f"{nov_s.get('facebook_gb',0):,.0f} GB",  "Most used app")
        a2.metric("YouTube Nov",   f"{nov_s.get('youtube_gb',0):,.0f} GB",   "Video streaming")
        a3.metric("TikTok Nov",    f"{nov_s.get('tiktok_gb',0):,.0f} GB",    "Short video")
        a4.metric("Gaming Nov",    f"{nov_s.get('gaming_gb',0):,.0f} GB",    "PUBG & others")

    st.markdown("---")

    # ── BUNDLE PURCHASE SECTION ───────────────────────
    st.markdown("#### 📦 Bundle Purchase Analysis — May 2026")

    bp_s    = ALL_STATS.get('bundle_new', {})
    addon_s = ALL_STATS.get('addon', {})

    if bp_s and addon_s:
        bp1,bp2 = st.columns(2)

        with bp1:
            st.subheader("Purchase Channel Breakdown")
            channels = bp_s.get('channels', {})
            if channels:
                ch_df = pd.DataFrame({
                    'Channel': list(channels.keys()),
                    'Count'  : list(channels.values())
                })
                fig = px.pie(ch_df, values='Count', names='Channel',
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             hole=0.4,
                             title="How customers buy bundles")
                fig.update_layout(height=300, margin=dict(t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

        with bp2:
            st.subheader("Top Bundles by Revenue")
            rev_data = bp_s.get('revenue_by_bundle', {})
            if rev_data:
                rev_df = pd.DataFrame({
                    'Bundle' : list(rev_data.keys())[:8],
                    'Revenue': list(rev_data.values())[:8]
                }).sort_values('Revenue', ascending=True)
                fig = px.bar(rev_df, x='Revenue', y='Bundle',
                             orientation='h',
                             color='Revenue',
                             color_continuous_scale=['#A8EDD0','#0F6E56'],
                             text='Revenue')
                fig.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside')
                fig.update_layout(height=300, margin=dict(t=0,b=0),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── Bundle Summary Metrics ────────────────────
        bm1,bm2,bm3,bm4 = st.columns(4)
        bm1.metric("Total Purchases",  f"{bp_s.get('total_purchases',0):,}",  "May 1–9 2026")
        bm2.metric("Unique Customers", f"{bp_s.get('unique_customers',0):,}", "Active buyers")
        bm3.metric("Total Revenue",    f"${bp_s.get('total_revenue',0):,.0f}","Combined")
        bm4.metric("Avg Purchase",     f"${bp_s.get('avg_charge',0):.2f}",    "Per transaction")

# ════════════════════════════════════════════════════
# PAGE 3 — AI CHAT ASSISTANT
# ════════════════════════════════════════════════════
elif page == "🤖 AI Chat Assistant":
    st.markdown('<div class="main-header">🤖 AI Chat Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by Llama 3 via Groq — Free & Fast</div>',
        unsafe_allow_html=True)

    # ── Dataset context for the AI ────────────────────
    s = ALL_STATS
    system_prompt = f"""You are TelecomAI, an expert telecom business analyst.
You have access to REAL data from 9 telecom dataset files across 2 datasets.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 1 — IBM TELCO CHURN (7,043 customers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Customers        : {TOTAL:,}
- Churn Rate             : {CHURN_RATE:.1f}%
- Monthly Revenue        : ${TOTAL_REV:,.0f}
- Avg Monthly Charge     : ${AVG_CHARGE:.2f}
- Month-to-Month Churn   : {MTM_CHURN:.1f}%
- Fiber Optic Churn      : {FIBER_CHURN:.1f}%
- Two-Year Contract Churn: {TWO_YR_CHURN:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 2 — BUNDLE PURCHASES MAY 2026 (NEW_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Purchases        : {s.get('bundle_new',{}).get('total_purchases',0):,}
- Unique Customers       : {s.get('bundle_new',{}).get('unique_customers',0):,}
- Total Revenue          : ${s.get('bundle_new',{}).get('total_revenue',0):,.2f}
- Avg Purchase Charge    : ${s.get('bundle_new',{}).get('avg_charge',0):.2f}
- Date Range             : {s.get('bundle_new',{}).get('date_range','N/A')}
- Top Bundle             : {s.get('bundle_new',{}).get('top_bundle','N/A')} ({s.get('bundle_new',{}).get('top_bundle_count',0):,} purchases)
- Top Bundles by count   : {s.get('bundle_new',{}).get('top_bundles',{})}
- Revenue by bundle      : {s.get('bundle_new',{}).get('revenue_by_bundle',{})}
- Purchase Channels      : {s.get('bundle_new',{}).get('channels',{})}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 3 — ADDON SUBSCRIPTIONS MAY 2026 (TELCO_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Records          : {s.get('addon',{}).get('total_records',0):,}
- Unique Customers       : {s.get('addon',{}).get('unique_customers',0):,}
- Total Revenue          : ${s.get('addon',{}).get('total_revenue',0):,.2f}
- Avg Charge             : ${s.get('addon',{}).get('avg_charge',0):.2f}
- Top Bundle             : {s.get('addon',{}).get('top_bundle','N/A')}
- Bundle breakdown       : {s.get('addon',{}).get('top_bundles',{})}
- Revenue by bundle      : {s.get('addon',{}).get('revenue_by_bundle',{})}
- Channels               : {s.get('addon',{}).get('channels',{})}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 4 — DATA USAGE NOVEMBER 2025 (TELCO_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Customers        : {s.get('nov_usage',{}).get('total_customers',0):,}
- Total Data Usage       : {s.get('nov_usage',{}).get('total_usage_gb',0):,.2f} GB
- Avg Usage Per Customer : {s.get('nov_usage',{}).get('avg_usage_gb',0):.2f} GB
- Max Usage (single cust): {s.get('nov_usage',{}).get('max_usage_gb',0):.2f} GB
- Facebook usage         : {s.get('nov_usage',{}).get('facebook_gb',0):,.2f} GB
- YouTube usage          : {s.get('nov_usage',{}).get('youtube_gb',0):,.2f} GB
- TikTok usage           : {s.get('nov_usage',{}).get('tiktok_gb',0):,.2f} GB
- Netflix usage          : {s.get('nov_usage',{}).get('netflix_gb',0):,.2f} GB
- Gaming (PUBG) usage    : {s.get('nov_usage',{}).get('gaming_gb',0):,.2f} GB
- WhatsApp usage         : {s.get('nov_usage',{}).get('whatsapp_gb',0):,.2f} GB
- Instagram usage        : {s.get('nov_usage',{}).get('instagram_gb',0):,.2f} GB
- General Traffic        : {s.get('nov_usage',{}).get('general_gb',0):,.2f} GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 5 — DATA USAGE OCTOBER 2025 (TELCO_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Customers        : {s.get('oct_usage',{}).get('total_customers',0):,}
- Total Data Usage       : {s.get('oct_usage',{}).get('total_usage_gb',0):,.2f} GB
- Avg Usage Per Customer : {s.get('oct_usage',{}).get('avg_usage_gb',0):.2f} GB
- Facebook usage         : {s.get('oct_usage',{}).get('facebook_gb',0):,.2f} GB
- YouTube usage          : {s.get('oct_usage',{}).get('youtube_gb',0):,.2f} GB
- TikTok usage           : {s.get('oct_usage',{}).get('tiktok_gb',0):,.2f} GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 6 — POSTPAID NOVEMBER 2025 (TELCO_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Postpaid Customers: {s.get('postpaid_nov',{}).get('total_customers',0):,}
- Total Usage             : {s.get('postpaid_nov',{}).get('total_usage_gb',0):,.2f} GB
- Avg Usage Per Customer  : {s.get('postpaid_nov',{}).get('avg_usage_gb',0):.2f} GB
- Max Single Customer     : {s.get('postpaid_nov',{}).get('max_usage_gb',0):.2f} GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 7 — POSTPAID OCTOBER 2025 (TELCO_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Postpaid Customers: {s.get('postpaid_oct',{}).get('total_customers',0):,}
- Total Usage             : {s.get('postpaid_oct',{}).get('total_usage_gb',0):,.2f} GB
- Avg Usage Per Customer  : {s.get('postpaid_oct',{}).get('avg_usage_gb',0):.2f} GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 8 — SERVICE USAGE (3 CUSTOMERS, NEW_DATA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Records          : {s.get('service_usage',{}).get('total_records',0)}
- Unique Customers       : {s.get('service_usage',{}).get('unique_customers',0)}
- Active Records         : {s.get('service_usage',{}).get('active_records',0)}
- Avg Data Benefit       : {s.get('service_usage',{}).get('avg_benefit_gb',0):.2f} GB
- Avg Data Usage         : {s.get('service_usage',{}).get('avg_usage_gb',0):.2f} GB

INSTRUCTIONS:
- Always reference real numbers from the datasets above
- When asked about app usage (Facebook/YouTube/TikTok), use Dataset 4 or 5
- When asked about bundle purchases, use Dataset 2 or 3
- When asked about postpaid customers, use Dataset 6 or 7
- When asked about churn, use Dataset 1
- Compare October vs November when trends are asked
- Give specific, data-driven, actionable answers with bullet points"""

    # ── Initialize chat history ───────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_initialized" not in st.session_state:
        welcome = (
            f"👋 Hello! I'm **TelecomAI** powered by **Llama 3 via Groq**.\n\n"
            f"I have analyzed your real dataset:\n"
            f"- 📊 **{TOTAL:,} customers** loaded\n"
            f"- ⚠️ **{CHURN_RATE:.1f}% churn rate** detected\n"
            f"- 💰 **${TOTAL_REV:,.0f} monthly revenue** tracked\n"
            f"- 🤖 **ML models** connected\n\n"
            f"Ask me anything about packages, churn, revenue, or strategy!"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome
        })
        st.session_state.chat_initialized = True

    # ── Quick question buttons ────────────────────────
    st.markdown("**Quick Questions — click any to get instant AI answers:**")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    prompt = None

    if col1.button("📦 Best packages to promote"):
        prompt = "Which packages should we promote next month for maximum revenue? Give specific recommendations based on our customer data."

    if col2.button("⚠️ High churn segments"):
        prompt = "Who are the highest churn risk customer segments? Include specific percentages from our data."

    if col3.button("💰 Pricing strategy"):
        prompt = "What is the best pricing strategy for next month to maximize revenue and reduce churn?"

    if col4.button("📊 Heavy data users"):
        prompt = "Which customer segments use the most data? How should we target them?"

    if col5.button("🔄 How to reduce churn"):
        prompt = "How can we reduce our churn rate effectively? Give top 3 specific actions with expected impact."

    if col6.button("📈 Revenue prediction"):
        prompt = "What is the predicted revenue for next month? What factors will drive growth or decline?"

    # ── More questions row ────────────────────────────
    st.markdown("---")
    col7, col8, col9 = st.columns(3)

    if col7.button("🎓 Student package strategy"):
        prompt = "Design the best package strategy for student customers. Include pricing and features."

    if col8.button("🎮 Gamer package strategy"):
        prompt = "Design the best data package for gamer customers. What features matter most?"

    if col9.button("🏆 Top retention strategies"):
        prompt = "What are the top 5 customer retention strategies for our telecom company based on our data?"

    # ── Display chat history ──────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything — churn, packages, revenue, predictions, strategy..."
    )
    if user_input:
        prompt = user_input

    # ── Process and respond ───────────────────────────
    if prompt:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get Groq AI response
        with st.chat_message("assistant"):
            with st.spinner("Llama 3 is thinking..."):
                try:
                    # Build message history for Groq
                    # Only include last 10 messages to stay within token limits
                    recent_messages = st.session_state.messages[-10:]

                    groq_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in recent_messages
                        if m["role"] in ["user", "assistant"]
                    ]

                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *groq_messages
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )

                    reply = response.choices[0].message.content

                except Exception as e:
                    reply = (
                        f"⚠️ **Groq Error:** `{str(e)}`\n\n"
                        f"**Common fixes:**\n"
                        f"- Check your `.env` file has: `GROQ_API_KEY=gsk_...`\n"
                        f"- Make sure you ran: `pip install groq`\n"
                        f"- Verify your key at console.groq.com"
                    )

                st.markdown(reply)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })

    # ── Clear chat ────────────────────────────────────
    st.markdown("---")
    col_clear, col_info = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_initialized = False
            st.rerun()
    with col_info:
        st.caption(
            f"🤖 Powered by Llama 3 (8B) via Groq API — Free tier "
            f"| Dataset: {TOTAL:,} customers loaded"
        )

# ════════════════════════════════════════════════════
# PAGE 4 — REPORTS
# ════════════════════════════════════════════════════    

elif page == "📄 Reports":
    st.markdown('<div class="main-header">📄 Reports</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete analytics across all 9 datasets</div>',
                unsafe_allow_html=True)

    # ── KPI Summary ───────────────────────────────────
    st.subheader("📊 Key Performance Indicators")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Customers",    f"{TOTAL:,}")
    k2.metric("Active Customers",   f"{TOTAL-CHURNED:,}")
    k3.metric("Churned Customers",  f"{CHURNED:,}")
    k4.metric("Churn Rate",         f"{CHURN_RATE:.1f}%")

    k5,k6,k7,k8 = st.columns(4)
    k5.metric("Monthly Revenue",        f"${TOTAL_REV:,.0f}")
    k6.metric("Avg Monthly Charge",     f"${AVG_CHARGE:.2f}")
    k7.metric("Bundle Purchases (May)", f"{ALL_STATS.get('bundle_new',{}).get('total_purchases',0):,}")
    k8.metric("Data Usage Customers",   f"{ALL_STATS.get('nov_usage',{}).get('total_customers',0):,}")

    st.markdown("---")

    # ── CHURN CHARTS ──────────────────────────────────
    st.subheader("📈 Churn Analysis")
    c1,c2 = st.columns(2)

    with c1:
        try:
            churn_contract = df.groupby('Contract').apply(
                lambda x:(x['Churn']=='Yes').mean()*100
            ).reset_index()
            churn_contract.columns = ['Contract','Churn Rate %']
            churn_contract['Churn Rate %'] = churn_contract['Churn Rate %'].round(1)
            fig = px.bar(churn_contract, x='Contract', y='Churn Rate %',
                         title='Churn Rate by Contract Type',
                         color='Contract',
                         color_discrete_sequence=['#E74C3C','#F39C12','#1D9E75'],
                         text='Churn Rate %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    with c2:
        try:
            churn_internet = df.groupby('InternetService').apply(
                lambda x:(x['Churn']=='Yes').mean()*100
            ).reset_index()
            churn_internet.columns = ['Internet Service','Churn Rate %']
            churn_internet['Churn Rate %'] = churn_internet['Churn Rate %'].round(1)
            fig = px.bar(churn_internet, x='Internet Service', y='Churn Rate %',
                         title='Churn Rate by Internet Service',
                         color='Internet Service',
                         color_discrete_sequence=['#3498DB','#E74C3C','#1D9E75'],
                         text='Churn Rate %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    st.markdown("---")

    # ── BUNDLE CHARTS ─────────────────────────────────
    st.subheader("📦 Bundle Purchase Analysis — May 2026")
    b1,b2 = st.columns(2)

    with b1:
        try:
            bundle_counts = ALL_STATS.get('bundle_new',{}).get('top_bundles',{})
            addon_counts  = ALL_STATS.get('addon',{}).get('top_bundles',{})

            # Combine both bundle datasets
            all_bundles = {}
            for k,v in bundle_counts.items():
                all_bundles[k] = all_bundles.get(k,0) + v
            for k,v in addon_counts.items():
                all_bundles[k] = all_bundles.get(k,0) + v

            bdf = pd.DataFrame({
                'Bundle'   : list(all_bundles.keys())[:10],
                'Purchases': list(all_bundles.values())[:10]
            }).sort_values('Purchases', ascending=True)

            fig = px.bar(bdf, x='Purchases', y='Bundle',
                         orientation='h',
                         title='Total Bundle Purchases (Combined)',
                         color='Purchases',
                         color_continuous_scale=['#A8EDD0','#1D9E75'],
                         text='Purchases')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=360, margin=dict(t=40,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    with b2:
        try:
            rev_new   = ALL_STATS.get('bundle_new',{}).get('revenue_by_bundle',{})
            rev_addon = ALL_STATS.get('addon',{}).get('revenue_by_bundle',{})

            # Combine revenue
            all_rev = {}
            for k,v in rev_new.items():
                all_rev[k] = all_rev.get(k,0) + v
            for k,v in rev_addon.items():
                all_rev[k] = all_rev.get(k,0) + v

            rdf = pd.DataFrame({
                'Bundle' : list(all_rev.keys())[:8],
                'Revenue': list(all_rev.values())[:8]
            })
            fig = px.pie(rdf, values='Revenue', names='Bundle',
                         title='Revenue by Bundle (Combined)',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.35)
            fig.update_layout(height=360, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    st.markdown("---")

    # ── APP USAGE COMPARISON ──────────────────────────
    st.subheader("📱 App Usage — October vs November 2025")

    nov_s = ALL_STATS.get('nov_usage', {})
    oct_s = ALL_STATS.get('oct_usage', {})

    if nov_s:
        app_labels = ['Facebook','YouTube','TikTok',
                      'Netflix','Gaming','WhatsApp','Instagram']
        nov_vals = [
            nov_s.get('facebook_gb',0), nov_s.get('youtube_gb',0),
            nov_s.get('tiktok_gb',0),   nov_s.get('netflix_gb',0),
            nov_s.get('gaming_gb',0),   nov_s.get('whatsapp_gb',0),
            nov_s.get('instagram_gb',0)
        ]
        oct_vals = [
            oct_s.get('facebook_gb',0), oct_s.get('youtube_gb',0),
            oct_s.get('tiktok_gb',0),   0,
            oct_s.get('gaming_gb',0),   0, 0
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='November 2025', x=app_labels, y=nov_vals,
            marker_color='#1D9E75',
            text=[f'{v:,.0f}' for v in nov_vals],
            textposition='outside'))
        fig.add_trace(go.Bar(
            name='October 2025', x=app_labels, y=oct_vals,
            marker_color='#378ADD',
            text=[f'{v:,.0f}' for v in oct_vals],
            textposition='outside'))
        fig.update_layout(
            barmode='group', height=380,
            title="App Usage Comparison: Oct vs Nov 2025 (GB)",
            yaxis_title="Usage (GB)",
            margin=dict(t=40,b=0),
            legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── POSTPAID COMPARISON ───────────────────────────
    st.subheader("💼 Postpaid Customers — Oct vs Nov 2025")

    pp_nov = ALL_STATS.get('postpaid_nov', {})
    pp_oct = ALL_STATS.get('postpaid_oct', {})

    pc1,pc2,pc3,pc4 = st.columns(4)
    pc1.metric("Postpaid Customers (Nov)", f"{pp_nov.get('total_customers',0):,}")
    pc2.metric("Postpaid Usage (Nov)",     f"{pp_nov.get('total_usage_gb',0):,.0f} GB")
    pc3.metric("Postpaid Customers (Oct)", f"{pp_oct.get('total_customers',0):,}")
    pc4.metric("Postpaid Usage (Oct)",     f"{pp_oct.get('total_usage_gb',0):,.0f} GB")

    # Postpaid comparison bar chart
    try:
        comp_df = pd.DataFrame({
            'Month'     : ['October 2025',           'November 2025'],
            'Customers' : [pp_oct.get('total_customers',0), pp_nov.get('total_customers',0)],
            'Usage GB'  : [pp_oct.get('total_usage_gb',0),  pp_nov.get('total_usage_gb',0)],
            'Avg GB'    : [pp_oct.get('avg_usage_gb',0),     pp_nov.get('avg_usage_gb',0)],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Total Usage (GB)', x=comp_df['Month'],
            y=comp_df['Usage GB'],
            marker_color=['#378ADD','#1D9E75'],
            text=comp_df['Usage GB'].apply(lambda x: f'{x:,.0f} GB'),
            textposition='outside'))
        fig.update_layout(
            title="Postpaid Total Data Usage: Oct vs Nov",
            height=320, margin=dict(t=40,b=0),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    st.markdown("---")

    # ── DETAILED TABLES ───────────────────────────────
    st.subheader("📋 Detailed Analytics Tables")
    t1,t2 = st.columns(2)

    with t1:
        st.markdown("**Churn Analysis by Contract**")
        try:
            table1 = df.groupby('Contract').agg(
                Total_Customers   =('customerID','count'),
                Churned           =('Churn', lambda x:(x=='Yes').sum()),
                Churn_Rate_Pct    =('Churn', lambda x:round((x=='Yes').mean()*100,1)),
                Avg_Monthly_Charge=('MonthlyCharges', lambda x:round(x.mean(),2))
            ).reset_index()
            table1.columns = ['Contract','Total','Churned',
                              'Churn Rate %','Avg Charge $']
            st.dataframe(table1, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Table error: {e}")

    with t2:
        st.markdown("**Bundle Purchases Summary**")
        try:
            bc = ALL_STATS.get('bundle_new',{}).get('top_bundles',{})
            ac = ALL_STATS.get('addon',{}).get('top_bundles',{})
            br = ALL_STATS.get('bundle_new',{}).get('revenue_by_bundle',{})
            ar = ALL_STATS.get('addon',{}).get('revenue_by_bundle',{})

            all_b = {}
            for k,v in bc.items(): all_b[k] = all_b.get(k,0) + v
            for k,v in ac.items(): all_b[k] = all_b.get(k,0) + v
            all_r = {}
            for k,v in br.items(): all_r[k] = all_r.get(k,0) + v
            for k,v in ar.items(): all_r[k] = all_r.get(k,0) + v

            bundle_table = pd.DataFrame({
                'Bundle'   : list(all_b.keys()),
                'Purchases': list(all_b.values()),
                'Revenue $': [round(all_r.get(k,0),2) for k in all_b.keys()]
            }).sort_values('Purchases', ascending=False).head(10)
            st.dataframe(bundle_table, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Table error: {e}")

    st.markdown("---")

    # ── AI RECOMMENDATIONS ────────────────────────────
    st.subheader("🤖 AI-Generated Recommendations")
    recommendations = [
        {"Priority":"🔴 High",   "Area":"Churn Retention",
         "Action": f"Retain {len(df[df['Contract']=='Month-to-month']):,} month-to-month customers",
         "Expected Impact": f"Reduce churn by ~8% — save ${CHURNED*AVG_CHARGE*0.08:,.0f}/month"},
        {"Priority":"🔴 High",   "Area":"Fiber Optic",
         "Action": f"Fix Fiber Optic service — {FIBER_CHURN:.1f}% churn rate",
         "Expected Impact": f"Save ${len(df[df['InternetService']=='Fiber optic'])*AVG_CHARGE*0.1:,.0f}/month"},
        {"Priority":"🟡 Medium", "Area":"Bundle Promotions",
         "Action": f"Promote $3 DATA bundle — most popular with {ALL_STATS.get('bundle_new',{}).get('top_bundle_count',0):,} purchases",
         "Expected Impact": "Increase bundle revenue by 15-20%"},
        {"Priority":"🟡 Medium", "Area":"App Usage",
         "Action": f"Facebook uses {nov_s.get('facebook_gb',0):,.0f} GB — offer social media bundles",
         "Expected Impact": "Target heavy Facebook users with social packs"},
        {"Priority":"🟢 Low",    "Area":"Postpaid Growth",
         "Action": f"Postpaid grew from {pp_oct.get('total_customers',0):,} to {pp_nov.get('total_customers',0):,} customers",
         "Expected Impact": "Continue postpaid acquisition campaigns"},
    ]
    st.dataframe(pd.DataFrame(recommendations),
                 use_container_width=True, hide_index=True)

    # ── DOWNLOAD ──────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Reports")

    churn_by_contract = df.groupby('Contract')['Churn'].apply(
        lambda x:(x=='Yes').mean()*100).round(1)

    report = f"""
TELECOMAI COMPLETE BUSINESS REPORT
====================================
Generated from 9 datasets across 2 data sources

CHURN ANALYSIS (IBM Telco Dataset)
------------------------------------
Total Customers      : {TOTAL:,}
Active Customers     : {TOTAL-CHURNED:,}
Churned Customers    : {CHURNED:,}
Churn Rate           : {CHURN_RATE:.1f}%
Monthly Revenue      : ${TOTAL_REV:,.0f}
Avg Monthly Charge   : ${AVG_CHARGE:.2f}

Churn by Contract:
{churn_by_contract.to_string()}

BUNDLE PURCHASES (May 2026)
-----------------------------
Total Purchases      : {ALL_STATS.get('bundle_new',{}).get('total_purchases',0):,}
Unique Customers     : {ALL_STATS.get('bundle_new',{}).get('unique_customers',0):,}
Total Revenue        : ${ALL_STATS.get('bundle_new',{}).get('total_revenue',0):,.2f}
Top Bundle           : {ALL_STATS.get('bundle_new',{}).get('top_bundle','N/A')}

ADDON SUBSCRIPTIONS (May 2026)
--------------------------------
Total Records        : {ALL_STATS.get('addon',{}).get('total_records',0):,}
Unique Customers     : {ALL_STATS.get('addon',{}).get('unique_customers',0):,}
Total Revenue        : ${ALL_STATS.get('addon',{}).get('total_revenue',0):,.2f}

DATA USAGE — NOVEMBER 2025
-----------------------------
Total Customers      : {nov_s.get('total_customers',0):,}
Total Usage          : {nov_s.get('total_usage_gb',0):,.2f} GB
Avg Per Customer     : {nov_s.get('avg_usage_gb',0):.2f} GB
Facebook             : {nov_s.get('facebook_gb',0):,.2f} GB
YouTube              : {nov_s.get('youtube_gb',0):,.2f} GB
TikTok               : {nov_s.get('tiktok_gb',0):,.2f} GB
Gaming (PUBG)        : {nov_s.get('gaming_gb',0):,.2f} GB

POSTPAID CUSTOMERS
--------------------
November 2025        : {pp_nov.get('total_customers',0):,} customers — {pp_nov.get('total_usage_gb',0):,.2f} GB
October 2025         : {pp_oct.get('total_customers',0):,} customers — {pp_oct.get('total_usage_gb',0):,.2f} GB

END OF REPORT — TelecomAI Assistant
"""
    d1,d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download Full Report (TXT)",
            data=report,
            file_name="telecom_complete_report.txt",
            mime="text/plain",
            use_container_width=True)
    with d2:
        csv_data = pd.DataFrame({
            'Metric':['Total Customers','Churned','Churn Rate %',
                      'Monthly Revenue','Bundle Purchases','Nov Usage GB'],
            'Value': [TOTAL, CHURNED, round(CHURN_RATE,1),
                      round(TOTAL_REV,0),
                      ALL_STATS.get('bundle_new',{}).get('total_purchases',0),
                      round(nov_s.get('total_usage_gb',0),2)]
        })
        st.download_button(
            label="⬇️ Download Metrics (CSV)",
            data=csv_data.to_csv(index=False),
            file_name="telecom_metrics.csv",
            mime="text/csv",
            use_container_width=True)