import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="👥",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: bold;
        color: #1F4E79; text-align: center; padding: 20px 0 5px 0;
    }
    .sub-title {
        font-size: 1.1rem; color: #555555;
        text-align: center; margin-bottom: 30px;
    }
    .metric-card {
        background: #D6E4F0; border-radius: 10px;
        padding: 15px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1F4E79; }
    .metric-label { font-size: 0.9rem; color: #555555; }
    .insight-box {
        background: #EBF5FB; border-left: 4px solid #1F4E79;
        padding: 10px 15px; border-radius: 5px; margin: 10px 0;
    }
    .predict-yes {
        background: #FFE0E0; border-radius: 10px;
        padding: 20px; text-align: center;
    }
    .predict-no {
        background: #D5F5D5; border-radius: 10px;
        padding: 20px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">👥 HR Attrition Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">IBM HR Analytics | Diana Sahira | github.com/Dodidee</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/human-resources.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pilih halaman:", [
    "📊 Overview",
    "🔍 EDA - Exploratory Analysis",
    "🤖 Predict Attrition",
    "📈 Model Performance"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload Dataset**")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("hr_attrition_model.pkl")
        features = joblib.load("hr_attrition_features.pkl")
        return model, features
    except:
        return None, None

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.sidebar.info("Sila upload file CSV untuk mula.")
    df = None

model, features = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.header("📊 Dataset Overview")

    if df is not None:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        leave = (df["Attrition"] == "Yes").sum()
        stay = (df["Attrition"] == "No").sum()
        rate = round(leave / total * 100, 1)

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Jumlah Pekerja</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{leave}</div><div class="metric-label">Pekerja Berhenti</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{stay:,}</div><div class="metric-label">Pekerja Stay</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{rate}%</div><div class="metric-label">Attrition Rate</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Taburan Attrition")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df["Attrition"].value_counts()
            colors = ["#DD8452", "#4C72B0"]
            bars = ax.bar(counts.index, counts.values, color=colors)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(val), ha='center', va='bottom', fontweight='bold')
            ax.set_xlabel("Attrition")
            ax.set_ylabel("Count")
            ax.set_title("Employee Attrition Count")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Maklumat Dataset")
            st.dataframe(df.head(10), width='stretch')


        st.markdown("---")
        st.subheader("Statistik Ringkas")
        st.dataframe(df.describe().round(2), width='stretch')

        st.markdown('<div class="insight-box">Dataset IBM HR Analytics mengandungi 1,470 pekerja dan 35 kolum. Hanya 16.1% pekerja yang berhenti — ini dipanggil <b>Class Imbalance</b>, sebab itu kita guna <b>Recall</b> sebagai metrik utama.</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ Sila upload dataset CSV di sidebar untuk melihat overview.")
        st.info("Dataset yang diperlukan: **WA_Fn-UseC_-HR-Employee-Attrition.csv** dari Kaggle")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA - Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")

    if df is not None:
        df_eda = df.copy()
        df_eda["Attrition_num"] = df_eda["Attrition"].map({"Yes": 1, "No": 0})

        tab1, tab2, tab3 = st.tabs(["Faktor Kerja", "Faktor Demografi", "Faktor Kepuasan"])

        # TAB 1: Faktor Kerja
        with tab1:
            st.subheader("Faktor Berkaitan Kerja")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.countplot(data=df_eda, x="OverTime", hue="Attrition", ax=ax)
                ax.set_title("Attrition by OverTime")
                ax.set_xlabel("OverTime")
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown('<div class="insight-box">Pekerja yang buat <b>OverTime</b> mempunyai kadar attrition jauh lebih tinggi — FAKTOR PALING KUAT!</div>', unsafe_allow_html=True)

            with col2:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.countplot(data=df_eda, y="Department", hue="Attrition", ax=ax)
                ax.set_title("Attrition by Department")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown('<div class="insight-box">Jabatan <b>Sales</b> mempunyai nisbah attrition paling tinggi berbanding saiz jabatan.</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(data=df_eda, y="JobRole", hue="Attrition", ax=ax)
            ax.set_title("Attrition by Job Role")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('<div class="insight-box"><b>Sales Representative</b> dan <b>Laboratory Technician</b> paling berisiko. Manager dan Research Director paling stabil.</div>', unsafe_allow_html=True)

        # TAB 2: Faktor Demografi
        with tab2:
            st.subheader("Faktor Demografi")

            col1, col2, col3 = st.columns(3)
            for col, ylabel, title, insight in [
                (col1, "MonthlyIncome", "Monthly Income", "Pekerja yang berhenti purata gaji ~RM4,800 vs RM6,800 yang stay. Beza hampir RM2,000!"),
                (col2, "Age", "Age", "Pekerja yang berhenti lebih muda (purata 33 tahun) berbanding yang stay (37 tahun)."),
                (col3, "YearsAtCompany", "Years at Company", "Pekerja yang berhenti berkhidmat lebih singkat (~5 tahun) vs yang stay (~7 tahun)."),
            ]:
                with col:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.barplot(data=df_eda, x="Attrition", y=ylabel, errorbar=None, ax=ax)
                    ax.set_title(f"Avg {title} vs Attrition")
                    ax.set_xlabel("Attrition")
                    ax.set_ylabel(title)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # TAB 3: Faktor Kepuasan
        with tab3:
            st.subheader("Faktor Kepuasan & Gaya Hidup")

            col1, col2 = st.columns(2)
            factors = [
                ("JobSatisfaction", "Job Satisfaction (1=Low, 4=High)", "Kepuasan kerja rendah (1): Attrition rate 23% vs 11% (kepuasan tinggi)."),
                ("WorkLifeBalance", "Work Life Balance (1=Low, 4=High)", "Work Life Balance teruk (1): Attrition rate 31% — paling tinggi!"),
            ]
            for (col, (factor, xlabel, ins)) in zip([col1, col2], factors):
                with col:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.barplot(data=df_eda, x=factor, y="Attrition_num", errorbar=None, ax=ax)
                    ax.set_title(f"Attrition Rate by {factor}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Attrition Rate")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            factors2 = [
                ("MaritalStatus", "Marital Status", "Pekerja Single mempunyai attrition rate 26% — lebih 2x berbanding Divorced (10%)."),
                ("BusinessTravel", "Business Travel", "Travel_Frequently mempunyai attrition rate 25% vs Non-Travel hanya 8%."),
            ]
            for (col, (factor, xlabel, ins)) in zip([col1, col2], factors2):
                with col:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.barplot(data=df_eda, x=factor, y="Attrition_num", errorbar=None, ax=ax)
                    ax.set_title(f"Attrition Rate by {factor}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Attrition Rate")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ Sila upload dataset CSV di sidebar untuk melihat EDA.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Attrition":
    st.header("🤖 Predict Attrition Pekerja")
    st.markdown("Masukkan maklumat pekerja di bawah untuk predict sama ada pekerja tersebut berisiko berhenti.")

    if model is None:
        st.error("⚠️ Model tidak dijumpai! Pastikan fail **hr_attrition_model.pkl** dan **hr_attrition_features.pkl** berada dalam folder yang sama dengan app ini.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Maklumat Peribadi")
            age = st.slider("Umur", 18, 60, 30)
            gender = st.selectbox("Jantina", ["Male", "Female"])
            marital = st.selectbox("Status Perkahwinan", ["Single", "Married", "Divorced"])
            education = st.slider("Tahap Pendidikan (1-5)", 1, 5, 3)
            edu_field = st.selectbox("Bidang Pendidikan", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            distance = st.slider("Jarak dari Rumah (km)", 1, 30, 5)

        with col2:
            st.subheader("Maklumat Kerja")
            dept = st.selectbox("Jabatan", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Jawatan", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            job_level = st.slider("Tahap Jawatan (1-5)", 1, 5, 2)
            overtime = st.selectbox("OverTime", ["No", "Yes"])
            travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            years_company = st.slider("Tahun di Syarikat", 0, 40, 3)
            years_role = st.slider("Tahun dalam Role Semasa", 0, 18, 2)

        with col3:
            st.subheader("Maklumat Gaji & Kepuasan")
            monthly_income = st.number_input("Gaji Bulanan (RM)", 1000, 20000, 4000, 500)
            stock_option = st.slider("Stock Option Level (0-3)", 0, 3, 0)
            job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
            wlb = st.slider("Work Life Balance (1-4)", 1, 4, 2)
            job_inv = st.slider("Job Involvement (1-4)", 1, 4, 2)
            env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 2)
            rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 2)
            perf_rating = st.slider("Performance Rating (3-4)", 3, 4, 3)
            num_companies = st.slider("Bilangan Syarikat Sebelum Ini", 0, 9, 1)
            training = st.slider("Training Kali Tahun Ini", 0, 6, 2)
            years_promo = st.slider("Tahun Sejak Naik Pangkat", 0, 15, 1)
            years_manager = st.slider("Tahun dengan Manager Semasa", 0, 17, 2)
            total_working = st.slider("Jumlah Tahun Bekerja", 0, 40, 5)
            daily_rate = st.number_input("Daily Rate", 100, 1500, 500)
            hourly_rate = st.number_input("Hourly Rate", 30, 100, 60)
            monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 10000)
            pct_hike = st.slider("Peratus Kenaikan Gaji (%)", 11, 25, 14)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Sekarang!", width='stretch')

        if predict_btn:
            input_data = {
                "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance,
                "Education": education, "EnvironmentSatisfaction": env_sat,
                "HourlyRate": hourly_rate, "JobInvolvement": job_inv,
                "JobLevel": job_level, "JobSatisfaction": job_sat,
                "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies, "PercentSalaryHike": pct_hike,
                "PerformanceRating": perf_rating, "RelationshipSatisfaction": rel_sat,
                "StockOptionLevel": stock_option, "TotalWorkingYears": total_working,
                "TrainingTimesLastYear": training, "WorkLifeBalance": wlb,
                "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
                "YearsSinceLastPromotion": years_promo, "YearsWithCurrManager": years_manager,
                "BusinessTravel_Travel_Frequently": 1 if travel == "Travel_Frequently" else 0,
                "BusinessTravel_Travel_Rarely": 1 if travel == "Travel_Rarely" else 0,
                "Department_Research & Development": 1 if dept == "Research & Development" else 0,
                "Department_Sales": 1 if dept == "Sales" else 0,
                "EducationField_Life Sciences": 1 if edu_field == "Life Sciences" else 0,
                "EducationField_Marketing": 1 if edu_field == "Marketing" else 0,
                "EducationField_Medical": 1 if edu_field == "Medical" else 0,
                "EducationField_Other": 1 if edu_field == "Other" else 0,
                "EducationField_Technical Degree": 1 if edu_field == "Technical Degree" else 0,
                "Gender_Male": 1 if gender == "Male" else 0,
                "JobRole_Human Resources": 1 if job_role == "Human Resources" else 0,
                "JobRole_Laboratory Technician": 1 if job_role == "Laboratory Technician" else 0,
                "JobRole_Manager": 1 if job_role == "Manager" else 0,
                "JobRole_Manufacturing Director": 1 if job_role == "Manufacturing Director" else 0,
                "JobRole_Research Director": 1 if job_role == "Research Director" else 0,
                "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
                "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
                "JobRole_Sales Representative": 1 if job_role == "Sales Representative" else 0,
                "MaritalStatus_Married": 1 if marital == "Married" else 0,
                "MaritalStatus_Single": 1 if marital == "Single" else 0,
                "OverTime_Yes": 1 if overtime == "Yes" else 0,
            }

            input_df = pd.DataFrame([input_data])
            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[features]

            from sklearn.preprocessing import StandardScaler
            try:
                scaler = joblib.load("hr_attrition_scaler.pkl")
                input_scaled = scaler.transform(input_df)
            except:
                input_scaled = input_df.values

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            st.markdown("---")
            st.subheader("📋 Hasil Prediction")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="predict-yes">
                        <h2>⚠️ BERISIKO BERHENTI</h2>
                        <h3>Kebarangkalian: {probability[1]*100:.1f}%</h3>
                        <p>Pekerja ini berisiko tinggi untuk berhenti kerja.<br>
                        HR perlu ambil tindakan segera!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="predict-no">
                        <h2>✅ RENDAH RISIKO</h2>
                        <h3>Kebarangkalian berhenti: {probability[1]*100:.1f}%</h3>
                        <p>Pekerja ini berkemungkinan besar akan kekal berkhidmat.<br>
                        Teruskan program retention yang sedia ada.</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📊 Breakdown Kebarangkalian")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kebarangkalian Stay", f"{probability[0]*100:.1f}%")
            with col2:
                st.metric("Kebarangkalian Leave", f"{probability[1]*100:.1f}%")

            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ["#4C72B0", "#DD8452"]
            ax.barh(["Stay", "Leave"], [probability[0]*100, probability[1]*100], color=colors)
            ax.set_xlabel("Kebarangkalian (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate([probability[0]*100, probability[1]*100]):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perbandingan Model")
        perf_data = {
            "Model": ["Logistic Regression", "Random Forest"],
            "Recall (Leave)": ["64%", "26%"],
            "Precision (Leave)": ["30%", "56%"],
            "Accuracy": ["75%", "87%"],
            "Sesuai untuk HR?": ["✅ Ya", "❌ Kurang"]
        }
        st.dataframe(pd.DataFrame(perf_data), width='stretch', hide_index=True)
        st.markdown('<div class="insight-box"><b>Logistic Regression dipilih</b> kerana Recall 64% — lebih ramai pekerja berisiko berjaya dikesan berbanding Random Forest (26%).</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Confusion Matrix — Logistic Regression")
        cm_data = np.array([[196, 59], [14, 25]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Stay", "Leave"],
                    yticklabels=["Stay", "Leave"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Top 10 Feature Importance")

    features_imp = {
        "Faktor": ["OverTime_Yes", "MaritalStatus_Single", "BusinessTravel_Frequently",
                   "StockOptionLevel", "PerformanceRating", "Department_Sales",
                   "JobRole_Lab Technician", "JobInvolvement", "JobRole_Sales Rep", "Gender_Male"],
        "Importance": [1.50, 0.70, 0.52, 0.48, 0.44, 0.43, 0.39, 0.34, 0.33, 0.31]
    }
    df_imp = pd.DataFrame(features_imp).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1F4E79" if i == len(df_imp)-1 else "#4C72B0" for i in range(len(df_imp))]
    ax.barh(df_imp["Faktor"], df_imp["Importance"], color=colors)
    ax.set_title("Top 10 Feature Importance (Logistic Regression)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight-box"><b>OverTime_Yes</b> adalah faktor paling dominan (1.50) — jauh mengatasi faktor kedua MaritalStatus_Single (0.70). HR perlu segera tangani isu overtime!</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Metrik Penjelasan")
    metrics = {
        "Metrik": ["Recall", "Precision", "Accuracy", "Confusion Matrix"],
        "Soalan": [
            "Berapa % pekerja berisiko yang berjaya dikesan?",
            "Berapa % prediction 'berisiko' yang betul-betul berisiko?",
            "Berapa % keseluruhan yang betul?",
            "Tunjukkan semua 4 kemungkinan prediction sekaligus"
        ],
        "Nilai (LR)": ["64%", "30%", "75%", "196/59/14/25"],
        "Kepentingan": ["⭐⭐⭐ Paling penting", "⭐⭐ Penting", "⭐ Mengelirukan (imbalanced)", "⭐⭐⭐ Gambaran lengkap"]
    }
    st.dataframe(pd.DataFrame(metrics), width='stretch', hide_index=True)
