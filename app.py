from xhtml2pdf import pisa
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import datetime
import os
import joblib
import shap
import uuid
import csv
from PIL import Image
import qrcode
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import GridSearchCV

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="wide", initial_sidebar_state="expanded")

# Font paths
FONT_PATH_BN = "fonts/Noto Sans.ttf"
FONT_PATH_EN = "fonts/DejaVu Sans.ttf"
BASE_DIR = os.path.dirname(__file__)
REPORT_CSV_PATH = os.path.join(BASE_DIR, "reports", "report_log.csv")
FEEDBACK_CSV_PATH = os.path.join(BASE_DIR, "feedback", "feedback.csv")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
ADMIN_PHOTO_PATH = os.path.join(BASE_DIR, "admin_photo.jpg")
ADMIN_INFO = "Reg. No : 2422228, M.Tech (CSE), NIT, Silchar"

# Multilingual dictionary for UI & messages
LANGS = {
    "en": {
        "title": "AI-Driven Early Diabetes Detection",
        "language_label": "Select Language",
        "nav_prediction": "Prediction",
        "nav_shap": "SHAP Explanation",
        "nav_health_tips": "Health Tips",
        "nav_pdf_report": "Download Report",
        "nav_feedback": "Feedback",
        "Admin Panel": "অ্যাডমিন প্যানেল",
        "input_pregnancies": "Number of Pregnancies (optional)",
        "input_glucose": "Glucose Level",
        "input_blood_pressure": "Blood Pressure",
        "input_skin_thickness": "Skin Thickness",
        "input_insulin": "Insulin Level",
        "input_bmi": "Body Mass Index (BMI)",
        "input_dpf": "Diabetes Pedigree Function",
        "input_age": "Age",
        "select_bmi_cat": "BMI Category",
        "bmi_categories": ["Normal", "Overweight", "Obese"],
        "predict_button": "Predict Diabetes Risk",
        "risk_high": "⚠️ High Risk of Diabetes",
        "risk_low": "✅ Low Risk of Diabetes",
        "top_factors": "Top Risk Factors",
        "no_prediction": "No prediction made yet.",
        "health_tip_title": "Personalized Health Tips",
        "pdf_generate": "Generate PDF Report",
        "feedback_title": "Send us your feedback",
        "feedback_name": "Your Name",
        "feedback_email": "Your Email",
        "feedback_message": "Your Message",
        "feedback_submit": "Submit Feedback",
        "feedback_thanks": "✅ Thank you for your feedback!",
        "error_api": "API request failed. Please try again.",
        "health_tips_high": [
            "- Avoid sugar and processed foods.",
            "- Exercise for at least 30 minutes daily.",
            "- Maintain a healthy weight.",
            "- Monitor your blood sugar regularly.",
            "- Consult your doctor regularly."
        ],
        "health_tips_low": [
            "- Continue your healthy lifestyle!",
            "- Stay active and hydrated.",
            "- Eat a balanced diet rich in fiber.",
            "- Avoid smoking and excess alcohol."
        ],
    },
    "bn": {
        "title": "কৃত্রিম বুদ্ধিমত্তা দ্বারা ডায়াবেটিস পূর্বাভাস",
        "language_label": "ভাষা নির্বাচন করুন",
        "nav_prediction": "পূর্বাভাস",
        "nav_shap": "SHAP ব্যাখ্যা",
        "nav_health_tips": "স্বাস্থ্য পরামর্শ",
        "nav_pdf_report": "রিপোর্ট ডাউনলোড করুন",
        "nav_feedback": "প্রতিক্রিয়া",
        "input_pregnancies": "গর্ভধারণের সংখ্যা (ঐচ্ছিক)",
        "input_glucose": "গ্লুকোজ লেভেল",
        "input_blood_pressure": "রক্তচাপ",
        "input_skin_thickness": "ত্বকের পুরুত্ব",
        "input_insulin": "ইনসুলিন লেভেল",
        "input_bmi": "বডি মাস ইনডেক্স (BMI)",
        "input_dpf": "ডায়াবেটিস পেডিগ্রি ফাংশন",
        "input_age": "বয়স",
        "select_bmi_cat": "BMI বিভাগ",
        "bmi_categories": ["স্বাভাবিক", "অতিরিক্ত ওজন", "মোটা"],
        "predict_button": "ডায়াবেটিস ঝুঁকি অনুমান করুন",
        "risk_high": "⚠️ উচ্চ ডায়াবেটিস ঝুঁকি",
        "risk_low": "✅ কম ডায়াবেটিস ঝুঁকি",
        "top_factors": "শীর্ষ ঝুঁকি কারণ",
        "no_prediction": "কোন পূর্বাভাস করা হয়নি।",
        "health_tip_title": "ব্যক্তিগতকৃত স্বাস্থ্য পরামর্শ",
        "pdf_generate": "PDF রিপোর্ট তৈরি করুন",
        "feedback_title": "আপনার প্রতিক্রিয়া পাঠান",
        "feedback_name": "আপনার নাম",
        "feedback_email": "আপনার ইমেইল",
        "feedback_message": "আপনার বার্তা",
        "feedback_submit": "প্রেরণ করুন",
        "feedback_thanks": "✅ আপনার প্রতিক্রিয়ার জন্য ধন্যবাদ!",
        "error_api": "API অনুরোধ ব্যর্থ হয়েছে। অনুগ্রহ করে পুনরায় চেষ্টা করুন।",
        "health_tips_high": [
            "- চিনিযুক্ত এবং প্রক্রিয়াজাত খাবার এড়িয়ে চলুন।",
            "- প্রতিদিন অন্তত ৩০ মিনিট ব্যায়াম করুন।",
            "- সুস্থ ওজন বজায় রাখুন।",
            "- নিয়মিত রক্তের চিনির মাত্রা পরীক্ষা করুন।",
            "- নিয়মিত ডাক্তারের সাথে পরামর্শ করুন।"
        ],
        "health_tips_low": [
            "- আপনার সুস্থ জীবনযাপন অব্যাহত রাখুন!",
            "- সক্রিয় থাকুন এবং পর্যাপ্ত পানি পান করুন।",
            "- ফাইবার সমৃদ্ধ সুষম খাদ্য গ্রহণ করুন।",
            "- ধূমপান এবং অতিরিক্ত মদ্যপান এড়িয়ে চলুন।"
        ],
    }
}

# Load model from JSON instead of pickle
model = XGBClassifier()
model.load_model("backend/models/xgb_model.json")

# Load the scaler as usual
scaler = joblib.load("backend/models/scaler.pkl")
explainer = shap.Explainer(model)



# Language selection
lang_code = st.sidebar.selectbox("🌐 " + LANGS["en"]["language_label"], options=["en", "bn"], index=0)
T = lambda key: LANGS[lang_code].get(key, key)

# Initialize session state variables
for key in ['prediction_result', 'top_features', 'inputs']:
    if key not in st.session_state:
        st.session_state[key] = None


def generate_pdf(data, prediction_result, top_features, lang, app_title="Diabetes Risk Predictor"):
    os.makedirs("reports", exist_ok=True)
    report_id = f"DIA-{uuid.uuid4().hex[:8].upper()}"
    filename = f"{report_id}_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", filename)

    pdf_labels = {
        "title": "Diabetes Risk Predictor" if lang == "en" else "ডায়াবেটিস ঝুঁকি পূর্বাভাস",
        "report_id": "Report ID" if lang == "en" else "রিপোর্ট আইডি",
        "date": "Date" if lang == "en" else "তারিখ",
        "name": "Patient Name" if lang == "en" else "রোগীর নাম",
        "age": "Age" if lang == "en" else "বয়স",
        "gender": "Gender" if lang == "en" else "লিঙ্গ",
        "input_section": "Patient Input Data" if lang == "en" else "রোগীর ইনপুট তথ্য",
        "risk": "Diabetes Risk" if lang == "en" else "ডায়াবেটিস ঝুঁকি",
        "summary": "Scan QR to View Patient Summary" if lang == "en" else "QR স্ক্যান করে রিপোর্ট দেখুন",
        "top_factors": "Top Risk Factors" if lang == "en" else "শীর্ষ ঝুঁকির কারণ",
        "feature": "Feature" if lang == "en" else "ফিচার",
        "impact": "Impact (%)" if lang == "en" else "প্রভাব (%)",
        "qr_note": "(Includes name, age, gender, inputs, and risk)" if lang == "en" else "(রোগীর ইনপুট এবং ঝুঁকি তথ্য অন্তর্ভুক্ত)"
    }

    patient_name = data.get("Patient_Name", "N/A")
    patient_age = data.get("Patient_Age", "N/A")
    patient_gender = data.get("Gender", "N/A")
    test_date = data.get("Test_Date", datetime.date.today().strftime("%Y-%m-%d"))
    risk_percent = int(round(prediction_result.get("risk_percent", 0) * 100))
    risk_text = LANGS[lang]["risk_high"] if prediction_result['prediction'] == 1 else LANGS[lang]["risk_low"]

    # Create QR text
    qr_text = f"{pdf_labels['report_id']}: {report_id}\n{pdf_labels['name']}: {patient_name}\n{pdf_labels['age']}: {patient_age}\n{pdf_labels['gender']}: {patient_gender}\n{pdf_labels['date']}: {test_date}\n{pdf_labels['risk']}: {risk_percent}% - {risk_text}\n"
    for k, v in data.items():
        if k.startswith("BMI_Category") and v:
            qr_text += f"{k.replace('_', ' ')}: Yes\n"
        elif k not in ["Patient_Name", "Patient_Age", "Gender", "Test_Date"]:
            qr_text += f"{k}: {v}\n"

    qr = qrcode.make(qr_text)
    qr_path = os.path.join("reports", "qr_temp.png")
    qr.save(qr_path)

    with open(qr_path, "rb") as qr_file:
        qr_base64 = base64.b64encode(qr_file.read()).decode("utf-8")
    qr_image_tag = f'''
    <table style="margin-left:auto; margin-right:auto; margin-top:10px;">
      <tr>
        <td style="text-align:center;">
          <img src="data:image/png;base64,{qr_base64}" width="120" alt="QR Code">
        </td>
      </tr>
    </table>
    '''

    rows = ""
    for k, v in data.items():
        if k in ["Patient_Name", "Patient_Age", "Gender", "Test_Date"]:
            continue
        if k.startswith("BMI_Category"):
            label = k.replace("_", " ").replace("BMI Category ", "")
            v = "Yes" if v else "No"
        else:
            label = k
        rows += f"<tr><td>{label}</td><td>{v}</td></tr>"

    factors = "".join([
        f"<tr><td>{f['feature']}</td><td>{abs(f['impact']) * 100:.1f}%</td></tr>"
        for f in top_features
    ])

    html_content = f"""
    <html lang="{lang}">
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #222;
                font-size: 16px;
            }}
            h1 {{
                text-align: center;
                font-size: 26px;
            }}
            .header-line {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .info {{
                padding: 10px;
                margin-bottom: 20px;
                background-color: #eef;
            }}
            .block-label {{
                font-weight: bold;
                font-size: 18px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            .risk {{
                background-color: #fff3cd;
                padding: 10px;
                font-weight: bold;
                margin-top: 20px;
                font-size: 18px;
            }}
            .qr-note {{
                font-size: 12px;
                text-align: center;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>{pdf_labels['title']}</h1>
        <div class="header-line">{pdf_labels['report_id']}: {report_id} | {pdf_labels['date']}: {test_date}</div>
        <div class="info">
            <p><span class="block-label">{pdf_labels['name']}:</span> {patient_name}</p>
            <p><span class="block-label">{pdf_labels['age']}:</span> {patient_age}</p>
            <p><span class="block-label">{pdf_labels['gender']}:</span> {patient_gender}</p>
        </div>

        <h2>{pdf_labels['input_section']}</h2>
        <table>{rows}</table>

        <div class="risk">
            <span class="block-label">{pdf_labels['risk']}:</span> {risk_percent}% — {risk_text}
        </div>

        <h2>{pdf_labels['top_factors']}</h2>
        <table>
            <tr><th>{pdf_labels['feature']}</th><th>{pdf_labels['impact']}</th></tr>
            {factors}
        </table>

        <div class="qr-code">
            <p style="text-align:center;"><strong>{pdf_labels['summary']}</strong></p>
            {qr_image_tag}
            <p class="qr-note">{pdf_labels['qr_note']}</p>
        </div>
    </body>
    </html>
    """

    # Use xhtml2pdf to create the PDF
    with open(filepath, "wb") as f:
        pisa.CreatePDF(html_content, dest=f)

    # Logging to CSV
    csv_path = "reports/report_log.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Report_ID", "Date", "Name", "Age", "Gender", "Risk (%)", "Risk_Text"])
        writer.writerow([report_id, test_date, patient_name, patient_age, patient_gender, risk_percent, risk_text])

    # Cleanup
    if os.path.exists(qr_path):
        os.remove(qr_path)

    return filepath


def download_link(filename):
    import os, base64

    if not os.path.exists(filename):
        return ""

    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    download_name = os.path.basename(filename)

    html = f"""
        <a href="data:application/octet-stream;base64,{b64}"
           download="{download_name}"
           style="
               display: inline-block;
               padding: 10px 20px;
               background-color: transparent;
               color: #FF4B4B;  /* Streamlit red */
               font-weight: bold;
               text-align: center;
               text-decoration: none;
               border: 2px solid #FF4B4B;
               border-radius: 8px;
               font-size: 14px;
               margin-top: 10px;
           ">
           📥 Download PDF Report
        </a>
    """
    return html



def plot_shap_bar_with_shap(input_scaled, shap_vals, feature_names):
    expl = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=input_scaled[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=(8, 6))
    shap.plots.bar(expl, max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.clf()


# App layout
st.title(T("title"))
tabs = st.tabs([
    T("nav_prediction"),
    T("nav_shap"),
    T("nav_health_tips"),
    T("nav_pdf_report"),
    T("nav_feedback"),
    T("Admin Panel")
])

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age',
            'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese']

def tab_admin():
    import streamlit as st
    import os, base64
    import pandas as pd

    # === Constants ===
    ADMIN_INFO = "Reg. No : 2422228, M.Tech (CSE), NIT, Silchar"
    ADMIN_PHOTO_PATH = os.path.join(os.path.dirname(__file__), "admin_photo.jpg")
    REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "reports")
    REPORT_CSV_PATH = os.path.join(REPORTS_FOLDER, "report_log.csv")
    FEEDBACK_CSV_PATH = os.path.join(os.path.dirname(__file__), "feedback", "feedback.csv")

    # === Session State Initialization ===
    if "admin_password" not in st.session_state:
        st.session_state.admin_password = "admin123"
    if "admin_name" not in st.session_state:
        st.session_state.admin_name = "Super Admin"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "login_error" not in st.session_state:
        st.session_state.login_error = False
    if "reset_error" not in st.session_state:
        st.session_state.reset_error = ""
    if "reset_mode" not in st.session_state:
        st.session_state.reset_mode = None




    # === Utility Functions ===
    def login(password):
        st.session_state.logged_in = password == st.session_state.admin_password
        st.session_state.login_error = not st.session_state.logged_in

    def logout():
        st.session_state.logged_in = False
        st.session_state.login_error = False
        st.session_state.reset_mode = None

    def get_pdf_download_link(filepath):
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        filename = os.path.basename(filepath)
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download PDF</a>'

    def load_report_logs():
        return pd.read_csv(REPORT_CSV_PATH) if os.path.exists(REPORT_CSV_PATH) else pd.DataFrame()

    def load_feedback():
        return pd.read_csv(FEEDBACK_CSV_PATH) if os.path.exists(FEEDBACK_CSV_PATH) else pd.DataFrame()

    # === UI Header ===
    st.title("🔒 Admin Panel")
    if os.path.exists(ADMIN_PHOTO_PATH):
        with open(ADMIN_PHOTO_PATH, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
    else:
        img_base64 = ""

    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 16px;">
            <img src="data:image/jpg;base64,{img_base64}" style="width:70px;height:70px;border-radius:50%;border:2px solid #555;" />
            <div>
                <h4 style="margin-bottom:0;">{st.session_state.admin_name}</h4>
                <p style="margin-top:4px;color:gray;">{ADMIN_INFO}</p>
            </div>
        </div>
        <hr style="margin-top:10px;margin-bottom:10px;">
    """, unsafe_allow_html=True)

    # === Main Login Block ===
    if not st.session_state.logged_in and st.session_state.reset_mode is None:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("🔓 Login"):
            login(password)
            if st.session_state.logged_in:
                st.success("✅ Login successful.")
        if st.session_state.login_error:
            st.error("❌ Incorrect password.")


    # === Forgot/Reset Options ===
    if st.session_state.reset_mode is None and not st.session_state.logged_in:
        st.subheader("🔧 What do you want to do?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Password"):
                st.session_state.reset_mode = "password"
                st.session_state.transition_flag = True
        with col2:
            if st.button("👤 Edit Admin Info"):
                st.session_state.reset_mode = "info"
                st.session_state.transition_flag = True

    # === Reset Password Mode ===
    if st.session_state.reset_mode == "password":
        st.subheader("🔄 Reset Admin Password")
        old = st.text_input("Old Password", type="password")
        new1 = st.text_input("New Password", type="password")
        new2 = st.text_input("Confirm New Password", type="password")

        if st.button("🔁 Change Password"):
            if old != st.session_state.admin_password:
                st.session_state.reset_error = "❌ Old password is incorrect."
            elif new1 != new2:
                st.session_state.reset_error = "❌ Passwords do not match."
            elif len(new1) < 6:
                st.session_state.reset_error = "❌ Password must be at least 6 characters."
            else:
                st.session_state.admin_password = new1
                st.success("✅ Password updated. Please login again.")
                st.session_state.logged_in = False
                st.session_state.reset_mode = None
                st.session_state.reset_error = ""

        if st.session_state.reset_error:
            st.error(st.session_state.reset_error)

        if st.button("🔙 Back", key="back_from_password_reset"):
            st.session_state.reset_mode = None
            st.session_state.logged_in = False

    # === Admin Info Login First ===
    if st.session_state.reset_mode == "info" and not st.session_state.logged_in:
        st.subheader("🔐 Please login to edit admin info")
        pw = st.text_input("Enter Password", type="password")
        if st.button("🔓 Login to Edit Info"):
            login(pw)
            if st.session_state.logged_in:
                st.success("✅ Login successful.")
        if st.session_state.login_error:
            st.error("❌ Incorrect password.")
        if st.button("🔙 Back", key="back_from_info_login"):
            st.session_state.reset_mode = None

    # === Admin Info Edit After Login ===
    if st.session_state.reset_mode == "info" and st.session_state.logged_in:
        st.subheader("👤 Edit Admin Name & Photo")
        new_name = st.text_input("New Admin Name", value=st.session_state.admin_name)
        photo = st.file_uploader("Upload New Admin Photo", type=["jpg", "jpeg", "png"])

        if st.button("✅ Update Info"):
            if not new_name.strip():
                st.error("❌ Admin name cannot be empty.")
            else:
                st.session_state.admin_name = new_name.strip()
                if photo:
                    with open(ADMIN_PHOTO_PATH, "wb") as f:
                        f.write(photo.getbuffer())
                    st.success("✅ Admin photo updated.")
                st.success("✅ Info updated.")

        if st.button("🔙 Back", key="back_from_info_edit"):
            st.session_state.reset_mode = None
            st.session_state.logged_in = False

    # === Dashboard (Only After Login) ===
    if st.session_state.logged_in and st.session_state.reset_mode is None:
        st.sidebar.button("Logout", on_click=logout)
        st.title("🛠️ Admin Management Dashboard")
        tab = st.sidebar.radio("📁 Navigation", ["📄 Reports", "💬 Feedback"])

        if tab == "📄 Reports":
            st.subheader("📄 Generated Reports")
            logs = load_report_logs()
            if logs.empty:
                st.info("No reports found.")
            else:
                for _, row in logs.iterrows():
                    with st.expander(
                            f"📄 {row.get('Report_ID', 'N/A')} — {row.get('Name', 'N/A')} ({row.get('Date', 'N/A')})"):
                        st.markdown(f"""
                            **🧑 Name:** {row.get('Name', 'N/A')}  
                            **🎂 Age:** {row.get('Age', 'N/A')}  
                            **⚥ Gender:** {row.get('Gender', 'N/A')}  
                            **📊 Risk:** {row.get('Risk (%)', 'N/A')}% — {row.get('Risk_Text', 'N/A')}
                        """)
                        matched = next(
                            (f for f in os.listdir(REPORTS_FOLDER) if row.get("Report_ID") in f and f.endswith(".pdf")),
                            None)
                        if matched:
                            st.markdown(get_pdf_download_link(os.path.join(REPORTS_FOLDER, matched)),
                                        unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ PDF not found for this report.")
                st.download_button("📥 Download All Logs CSV", logs.to_csv(index=False).encode("utf-8"),
                                   "report_log.csv", "text/csv")

        elif tab == "💬 Feedback":
            st.subheader("💬 User Feedback")
            feedback_df = load_feedback()
            if feedback_df.empty:
                st.info("No feedback submitted yet.")
            else:
                st.dataframe(feedback_df)
                st.download_button("📥 Download Feedback CSV", feedback_df.to_csv(index=False).encode("utf-8"),
                                   "feedback.csv", "text/csv")

# Prediction tab
with tabs[0]:
    st.header(T("nav_prediction"))

    with st.form("input_form"):
        st.subheader("🧑 Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            patient_name = st.text_input("Patient Name *")
        with col2:
            patient_age = st.number_input("Patient Age *", max_value=120)
        with col3:
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])

        st.subheader("🩺 Health Metrics")
        col3, col4 = st.columns(2)
        with col3:
            pregnancies = st.number_input(T("input_pregnancies"), min_value=0, max_value=20, value=0)
            glucose = st.number_input(T("input_glucose"), min_value=0, max_value=300, value=120)
            skin_thickness = st.number_input(T("input_skin_thickness"), min_value=0, max_value=100, value=20)
            dpf = st.number_input(T("input_dpf"), min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        with col4:
            blood_pressure = st.number_input(T("input_blood_pressure"), min_value=0, max_value=200, value=70)
            insulin = st.number_input(T("input_insulin"), min_value=0, max_value=900, value=79)
            bmi = st.number_input(T("input_bmi"), min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
            bmi_cat = st.selectbox(T("select_bmi_cat"), options=T("bmi_categories"))

        submit = st.form_submit_button("🚀 " + T("predict_button"))

    # ✅ Validate required fields
    if submit:
        if not patient_name.strip():
            st.error("❌ Patient Name is required.")
        elif not patient_age:
            st.error("❌ Patient Age is required.")
        elif not gender:
            st.error("❌ Gender selection is required.")
        else:
            bmi_normal = 1 if bmi_cat == T("bmi_categories")[0] else 0
            bmi_overweight = 1 if bmi_cat == T("bmi_categories")[1] else 0
            bmi_obese = 1 if bmi_cat == T("bmi_categories")[2] else 0

            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                          bmi, dpf, patient_age,
                          bmi_normal, bmi_overweight, bmi_obese]

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            pred_proba = model.predict_proba(input_scaled)[0][1]
            pred_class = 1 if pred_proba >= 0.5 else 0

            st.session_state['prediction_result'] = {
                "prediction": pred_class,
                "risk_percent": pred_proba
            }

            st.session_state['inputs'] = {
                "Patient_Name": patient_name,
                "Patient_Age": patient_age,
                "Gender": gender,
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": patient_age,
                "BMI_Category_Normal": bmi_normal,
                "BMI_Category_Overweight": bmi_overweight,
                "BMI_Category_Obese": bmi_obese
            }

            shap_values = explainer.shap_values(input_scaled)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_vals_abs = np.abs(shap_vals)

            top_indices = np.argsort(shap_vals_abs[0])[::-1][:5]
            all_features = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age",
                "BMI_Category_Normal", "BMI_Category_Overweight", "BMI_Category_Obese"
            ]
            top_factors = [{"feature": all_features[i], "impact": shap_vals_abs[0][i]} for i in top_indices]
            st.session_state['top_features'] = top_factors

            if pred_class == 1:
                st.error(f"{T('risk_high')} ({int(pred_proba * 100)}%)")
            else:
                st.success(f"{T('risk_low')} ({int(pred_proba * 100)}%)")

            st.subheader(T("top_factors"))
            for item in top_factors:
                st.write(f"{item['feature']}: {item['impact'] * 100:.2f}%")

with tabs[1]:
    st.header(T("nav_shap"))
    if st.session_state['prediction_result'] is not None and st.session_state['inputs'] is not None:
        # Prepare input for SHAP explainer
        input_data = [st.session_state['inputs'][f] for f in features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        shap_values = explainer.shap_values(input_scaled)

        # Choose correct SHAP values if multiclass
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        if shap_vals.ndim == 1:
            shap_vals_exp = shap_vals.reshape(1, -1)
        else:
            shap_vals_exp = shap_vals

        # Exclude categorical BMI one-hot columns and custom features if needed
        exclude_features = ["BMI_Category_Overweight", "BMI_Category_Normal", "BMI_Category_Obese", "Sum_of_Two_Features"]

        filtered_features = []
        filtered_shap_vals = []

        for i, feat in enumerate(features):
            if feat not in exclude_features:
                filtered_features.append(feat)
                filtered_shap_vals.append(shap_vals_exp[0][i])

        filtered_shap_vals = np.array(filtered_shap_vals).reshape(1, -1)

        base_values = None
        if hasattr(explainer, "expected_value"):
            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                base_values = explainer.expected_value[1]
            else:
                base_values = explainer.expected_value

        expl = shap.Explanation(
            values=filtered_shap_vals[0],
            base_values=base_values,
            data=input_scaled[:, [features.index(f) for f in filtered_features]][0],
            feature_names=filtered_features,
        )

        plt.figure(figsize=(8, 6))
        shap.plots.bar(expl, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        st.info(T("no_prediction"))




# Health tips tab
with tabs[2]:
    st.header(T("nav_health_tips"))
    if st.session_state['prediction_result'] is not None:
        if st.session_state['prediction_result']['prediction'] == 1:
            tips = T("health_tips_high")
        else:
            tips = T("health_tips_low")
        for tip in tips:
            st.write(f"• {tip}")
    else:
        st.info(T("no_prediction"))

# PDF Report tab
with tabs[3]:
    st.header(T("nav_pdf_report"))
    if st.session_state['prediction_result'] is not None and st.session_state['inputs'] is not None:
        if st.button(T("pdf_generate")):
            filepath = generate_pdf(st.session_state['inputs'], st.session_state['prediction_result'], st.session_state['top_features'], lang_code)
            st.markdown(download_link(filepath), unsafe_allow_html=True)
    else:
        st.info(T("no_prediction"))

# Feedback tab
with tabs[4]:
    st.header(T("nav_feedback"))
    with st.form("feedback_form"):
        name = st.text_input(T("feedback_name"))
        email = st.text_input(T("feedback_email"))
        message = st.text_area(T("feedback_message"))
        submit = st.form_submit_button(T("feedback_submit"))

    if submit:
        # Save feedback to CSV
        feedback_file = "feedback.csv"
        feedback_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "email": email,
            "message": message
        }

        if os.path.exists(feedback_file):
            df_existing = pd.read_csv(feedback_file)
            df_updated = pd.concat([df_existing, pd.DataFrame([feedback_data])], ignore_index=True)
        else:
            df_updated = pd.DataFrame([feedback_data])

        df_updated.to_csv(feedback_file, index=False)

        st.success(T("feedback_thanks"))

with tabs[5]:
    tab_admin()
