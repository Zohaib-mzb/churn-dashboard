import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import shap
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background: Light Soft Blue */
    .stApp {
        background-color: #EBF3FA !important;
    }

    /* Sidebar Background: Deep Navy Blue */
    [data-testid="stSidebar"] {
        background-color: #0A192F !important;
    }

    /* Target MAIN APP content specifically (avoids affecting the sidebar text) */
    .stMain p, 
    .stMain span, 
    .stMain label, 
    .stMain div,
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] div {
        color: #0F172A !important;
    }

    /* Sidebar Text, Navigation Labels, Radio Buttons, and Subtext: Bright White & Light Gray */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }

    /* Sidebar Links (e.g. GitHub) */
    [data-testid="stSidebar"] a {
        color: #38BDF8 !important;
    }

    /* Metric Cards: Crisp White Card with Dark Contrast Text */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 18px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }

    /* Metric Label */
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Metric Value */
    [data-testid="stMetricValue"] {
        color: #0A192F !important;
    }

    /* Alert / Info / Success Boxes Text */
    [data-testid="stNotification"] p,
    [data-testid="stAlert"] p {
        color: #0F172A !important;
    }

    /* Section Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0A192F !important;
        font-weight: 700 !important;
    }

    /* Make top header background transparent so collapse button stays visible */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }

    /* Menu Toggle Button: Dark Navy when collapsed on main screen */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg {
        color: #0A192F !important;
        fill: #0A192F !important;
    }

    /* Menu Toggle Button: Bright White when sidebar is open */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Hide Streamlit Footer & Main Menu Branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw['TotalCharges'] = pd.to_numeric(
        df_raw['TotalCharges'], errors='coerce').fillna(0)
    df_raw['TenureGroup'] = pd.cut(
        df_raw['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
    )
    return df_raw

@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_processed():
    X_test  = pd.read_csv('X_test.csv')
    y_test  = pd.read_csv('y_test.csv').squeeze()
    X_train = pd.read_csv('X_train.csv')
    return X_test, y_test, X_train

df      = load_data()
model   = load_model()
X_test, y_test, X_train = load_processed()

# ── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Churn Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Overview",
         "🔍  Customer Segments",
         "🤖  Predict Churn"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"7,043 customers · 21 features")
    st.markdown("IBM Telco — Q3 California")
    st.markdown("---")
    st.markdown("**Built by**")
    st.markdown("Muhammad Zohaib")
    st.markdown("[GitHub](https://github.com/Zohaib-mzb)")

# ── HELPER: CHURN RATE BY COLUMN ────────────────────────────────
def churn_rate_by(col, exclude_val=None):
    temp = df.copy()
    if exclude_val:
        temp = temp[temp[col] != exclude_val]
    return temp.groupby(col, observed=True)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index().rename(columns={'Churn': 'ChurnRate'})

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if '🏠' in page:
    st.title("📊 Customer Churn Intelligence Dashboard")
    st.markdown("*Telco customer churn analysis IBM dataset · 7,043 customers*")
    st.markdown("---")

    # ── KPI METRICS ROW ─────────────────────────────────────────
    total     = len(df)
    churned   = (df['Churn'] == 'Yes').sum()
    retained  = total - churned
    churn_pct = churned / total * 100
    avg_charge_churned = df[df['Churn']=='Yes']['MonthlyCharges'].mean()
    avg_tenure_churned = df[df['Churn']=='Yes']['tenure'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",    f"{total:,}")
    c2.metric("Churned",            f"{churned:,}",
              delta=f"-{churn_pct:.1f}% of base", delta_color="inverse")
    c3.metric("Retained",           f"{retained:,}")
    c4.metric("Avg Charge (Churned)", f"${avg_charge_churned:.0f}/mo")
    c5.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.0f} months")

    st.markdown("---")

    # ── CHARTS ROW 1 ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Churn Rate")
        fig, ax = plt.subplots(figsize=(6, 6))
        colors  = ['#2ecc71', '#e74c3c']
        sizes   = [retained, churned]
        labels  = [f"Stayed\n{retained:,} ({100-churn_pct:.1f}%)",
                   f"Churned\n{churned:,} ({churn_pct:.1f}%)"]
        ax.pie(sizes, labels=labels, colors=colors,
               startangle=90, wedgeprops=dict(width=0.55),
               textprops={'fontsize': 12})
        ax.set_title("", pad=0)
        fig.patch.set_facecolor('none')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn by Contract Type")
        ct = churn_rate_by('Contract')
        ct = ct.sort_values('ChurnRate', ascending=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(ct['Contract'], ct['ChurnRate'],
                      color=['#e74c3c', '#f39c12', '#2ecc71'],
                      width=0.5, edgecolor='white')
        for bar, val in zip(bars, ct['ChurnRate']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center',
                    va='bottom', fontweight='bold')
        ax.set_ylabel('Churn Rate (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 55)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('none')
        st.pyplot(fig)
        plt.close()

    # ── KEY FINDINGS ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Key Business Findings")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("**📋 Contract Type**\n\nMonth-to-month customers "
                "churn at ~43% vs ~11% for two-year contracts. "
                "Contract length is the single strongest retention lever.")
    with f2:
        st.warning("**⏱️ First 12 Months**\n\nNew customers (0–12 months) "
                   "churn at nearly 48%. The critical retention window "
                   "is the first year of service.")
    with f3:
        st.error("**💳 Payment Method**\n\nElectronic check users churn "
                 "at ~45%. Customers on automatic payments churn at less "
                 "than half that rate.")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════════
elif '🔍' in page:
    st.title("🔍 Customer Segment Analysis")
    st.markdown("Explore churn rate across every customer segment.")
    st.markdown("---")

    # ── FILTER ──────────────────────────────────────────────────
    segment = st.selectbox(
        "Select segment to analyse:",
        ["Tenure Group", "Payment Method",
         "Internet Service", "Support Services",
         "Monthly Charges Range"]
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    if segment == "Tenure Group":
        data = churn_rate_by('TenureGroup')
        colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
        ax.set_title("Churn Rate by Tenure Group", fontsize=14, fontweight='bold')
        col_name = 'TenureGroup'

    elif segment == "Payment Method":
        data = churn_rate_by('PaymentMethod')
        colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
        ax.set_title("Churn Rate by Payment Method", fontsize=14, fontweight='bold')
        col_name = 'PaymentMethod'

    elif segment == "Internet Service":
        data = churn_rate_by('InternetService')
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax.set_title("Churn Rate by Internet Service Type", fontsize=14, fontweight='bold')
        col_name = 'InternetService'

    elif segment == "Monthly Charges Range":
        df['ChargesGroup'] = pd.cut(df['MonthlyCharges'],
                                     bins=[0, 30, 60, 90, 120],
                                     labels=['$0–30', '$30–60', '$60–90', '$90–120'])
        data = churn_rate_by('ChargesGroup')
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        ax.set_title("Churn Rate by Monthly Charges Range", fontsize=14, fontweight='bold')
        col_name = 'ChargesGroup'

    else:  # Support Services
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, svc in enumerate(['TechSupport', 'OnlineSecurity', 'OnlineBackup']):
            d = churn_rate_by(svc, exclude_val='No internet service')
            c_map = {'No': '#e74c3c', 'Yes': '#2ecc71'}
            clrs  = [c_map.get(v, '#3498db') for v in d[svc]]
            bars  = axes[i].bar(d[svc], d['ChurnRate'],
                                color=clrs, width=0.4, edgecolor='white')
            for bar, val in zip(bars, d['ChurnRate']):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.5,
                             f'{val:.1f}%', ha='center',
                             va='bottom', fontweight='bold')
            axes[i].set_title(svc, fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Churn Rate (%)' if i == 0 else '')
            axes[i].set_ylim(0, 55)
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        fig.suptitle("Support Services vs Churn Rate", fontsize=14, fontweight='bold')
        fig.patch.set_facecolor('none')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        data = None

    if data is not None:
        data = data.sort_values('ChurnRate', ascending=False)
        bars = ax.bar(data[col_name].astype(str),
                      data['ChurnRate'],
                      color=colors[:len(data)],
                      width=0.5, edgecolor='white')
        for bar, val in zip(bars, data['ChurnRate']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center',
                    va='bottom', fontweight='bold')
        ax.set_ylabel('Churn Rate (%)', fontsize=12)
        ax.set_ylim(0, 60)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('none')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── SEGMENT SUMMARY TABLE ────────────────────────────────────
    st.markdown("---")
    st.subheader("Full Churn Summary by Segment")

    summary_data = {
        'Segment': ['Month-to-Month Contract', '0–12 Month Tenure',
                    'Electronic Check', 'Fibre Optic Internet',
                    'No Tech Support', 'No Online Security',
                    'Two-Year Contract', '49–72 Month Tenure'],
        'Churn Rate': ['~43%', '~48%', '~45%', '~42%',
                       '~41%', '~42%', '~11%', '~6%'],
        'Risk Level': ['🔴 High', '🔴 High', '🔴 High', '🔴 High',
                       '🟡 Medium', '🟡 Medium', '🟢 Low', '🟢 Low'],
        'Action': ['Offer contract upgrade incentive',
                   'Intensive onboarding programme',
                   'Migrate to auto-pay with discount',
                   'Improve service quality / pricing',
                   'Upsell tech support bundle',
                   'Upsell security package',
                   'Maintain loyalty rewards',
                   'Ambassador / referral programme']
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CHURN
# ════════════════════════════════════════════════════════════════
else:
    st.title("🤖 Predict Customer Churn")
    st.markdown("Enter a customer's details to get a live churn prediction "
                "with AI-powered explanation.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Customer Profile")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges   = st.number_input("Total Charges ($)",
                                          min_value=0.0,
                                          value=float(tenure * monthly_charges))
        senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner         = st.selectbox("Has Partner",    ["No", "Yes"])
        dependents      = st.selectbox("Has Dependents", ["No", "Yes"])

    with col2:
        st.subheader("📡 Services")
        phone_service    = st.selectbox("Phone Service",    ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_backup    = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
        device_prot      = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
        tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💳 Account Info")
        contract         = st.selectbox("Contract Type",
                                        ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment          = st.selectbox("Payment Method",
                                        ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)",
                                         "Credit card (automatic)"])
        gender           = st.selectbox("Gender", ["Male", "Female"])
        multiple_lines   = st.selectbox("Multiple Lines",
                                        ["No", "Yes", "No phone service"])

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):

        # ── BUILD FEATURE ROW ───────────────────────────────────
        yn  = lambda v: 1 if v == 'Yes' else 0
        row = {
            'gender':                  1 if gender == 'Male' else 0,
            'SeniorCitizen':           yn(senior_citizen),
            'Partner':                 yn(partner),
            'Dependents':              yn(dependents),
            'tenure':                  tenure,
            'PhoneService':            yn(phone_service),
            'MultipleLines':           yn(multiple_lines),
            'OnlineSecurity':          yn(online_security),
            'OnlineBackup':            yn(online_backup),
            'DeviceProtection':        yn(device_prot),
            'TechSupport':             yn(tech_support),
            'StreamingTV':             yn(streaming_tv),
            'StreamingMovies':         yn(streaming_movies),
            'PaperlessBilling':        yn(paperless),
            'MonthlyCharges':          monthly_charges,
            'TotalCharges':            total_charges,
            'InternetService_DSL':     1 if internet_service == 'DSL' else 0,
            'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No':      1 if internet_service == 'No' else 0,
            'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
            'Contract_One year':       1 if contract == 'One year' else 0,
            'Contract_Two year':       1 if contract == 'Two year' else 0,
            'PaymentMethod_Bank transfer (automatic)':
                1 if payment == 'Bank transfer (automatic)' else 0,
            'PaymentMethod_Credit card (automatic)':
                1 if payment == 'Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check':
                1 if payment == 'Electronic check' else 0,
            'PaymentMethod_Mailed check':
                1 if payment == 'Mailed check' else 0,
        }

        # Align columns exactly with training data
        input_df = pd.DataFrame([row])
        for col in X_train.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_train.columns]

        # ── PREDICTION ──────────────────────────────────────────
        prob       = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        r1, r2 = st.columns([1, 2])

        with r1:
            st.subheader("Prediction Result")
            if prob >= 0.6:
                st.error(f"### 🔴 HIGH CHURN RISK\n**{prob*100:.1f}% probability**")
                st.markdown("This customer is likely to leave. "
                            "Immediate retention action recommended.")
            elif prob >= 0.35:
                st.warning(f"### 🟡 MEDIUM CHURN RISK\n**{prob*100:.1f}% probability**")
                st.markdown("Monitor this customer. "
                            "Consider a proactive check-in or offer.")
            else:
                st.success(f"### 🟢 LOW CHURN RISK\n**{prob*100:.1f}% probability**")
                st.markdown("Customer appears stable. "
                            "Focus on maintaining satisfaction.")

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 3))
            color = '#e74c3c' if prob >= 0.6 else '#f39c12' if prob >= 0.35 else '#2ecc71'
            ax.barh(['Risk'], [prob * 100], color=color,
                    height=0.4, edgecolor='white')
            ax.barh(['Risk'], [100 - prob * 100], left=[prob * 100],
                    color='#E5E7EB', height=0.4, edgecolor='white')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Churn Probability (%)')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title(f'Churn Probability: {prob*100:.1f}%', fontweight='bold')
            fig.patch.set_facecolor('none')
            st.pyplot(fig)
            plt.close()

        with r2:
            st.subheader("Why This Prediction? (SHAP Explanation)")
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(input_df)

            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(shap_vals, input_df,
                              plot_type='bar', max_display=10,
                              show=False)
            ax.set_title("Top Factors Driving This Prediction",
                         fontsize=13, fontweight='bold')
            fig.patch.set_facecolor('none')
            st.pyplot(fig)
            plt.close()

            # Top 3 reasons in plain English
            feature_impact = pd.Series(
                np.abs(shap_vals[0]),
                index=X_train.columns
            ).sort_values(ascending=False)

            st.markdown("**Top 3 factors for this customer:**")
            for i, (feat, val) in enumerate(feature_impact.head(3).items()):
                direction = "↑ increases" if shap_vals[0][
                    X_train.columns.tolist().index(feat)] > 0 else "↓ decreases"
                st.markdown(f"{i+1}. **{feat}** — {direction} churn risk")

    # ── MODEL INFO ──────────────────────────────────────────────
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Algorithm:** XGBoost Classifier
        **Training data:** 5,634 customers (80% of IBM Telco dataset)
        **Test data:** 1,409 customers (20% holdout)
        **Class imbalance handling:** scale_pos_weight
        **Explainability:** SHAP TreeExplainer

        *This model was trained on synthetic IBM data for demonstration
        purposes. Real production use requires validation on live customer
        data and review by domain experts.*
        """)