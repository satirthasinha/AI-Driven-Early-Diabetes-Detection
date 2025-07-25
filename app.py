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
    qr_image_tag = f'''<img src="data:image/png;base64,{qr_base64}" style="display:block; margin:auto; width:120px;" alt="QR Code">'''

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
            }}
            h1 {{
                text-align: center;
                font-size: 24px;
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
            }}
            .qr-note {{
                font-size: 12px;
                text-align: center;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{pdf_labels['title']}</h1>
        <div class="header-line">{pdf_labels['report_id']}: {report_id} | {pdf_labels['date']}: {test_date}</div>
        <div class="info">
            <p><strong>{pdf_labels['name']}:</strong> {patient_name}</p>
            <p><strong>{pdf_labels['age']}:</strong> {patient_age}</p>
            <p><strong>{pdf_labels['gender']}:</strong> {patient_gender}</p>
        </div>

        <h2>{pdf_labels['input_section']}</h2>
        <table>{rows}</table>

        <div class="risk">📊 {pdf_labels['risk']}: {risk_percent}% — {risk_text}</div>

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

    # Use xhtml2pdf instead of WeasyPrint
    with open(filepath, "wb") as f:
        pisa.CreatePDF(html_content, dest=f)

    # Log
    csv_path = "reports/report_log.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Report_ID", "Date", "Name", "Age", "Gender", "Risk (%)", "Risk_Text"])
        writer.writerow([report_id, test_date, patient_name, patient_age, patient_gender, risk_percent, risk_text])

    if os.path.exists(qr_path):
        os.remove(qr_path)

    return filepath



def download_link(filename):
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    download_name = os.path.basename(filename)
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_name}">📥 Download PDF Report</a>'



def plot_shap_bar_with_shap(input_scaled, shap_vals, feature_names):
    expl = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value[1] if hasattr(explainer, "expected_value") else None,
        data=input_scaled[0],
        feature_names=feature_names,
    )
    shap.plots.bar(expl, max_display=10)
    st.pyplot(bbox_inches='tight')

# App layout
st.title(T("title"))
tabs = st.tabs([
    T("nav_prediction"),
    T("nav_shap"),
    T("nav_health_tips"),
    T("nav_pdf_report"),
    T("nav_feedback")
])

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age',
            'BMI_Category_Normal', 'BMI_Category_Overweight', 'BMI_Category_Obese']

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
        # Prepare input for shap explainer
        input_data = [st.session_state['inputs'][f] for f in features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        shap_values = explainer.shap_values(input_scaled)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        if shap_vals.ndim == 1:
            shap_vals_exp = shap_vals.reshape(1, -1)
        else:
            shap_vals_exp = shap_vals

        # Features to exclude from SHAP plot
        exclude_features = ["BMI_Category_Overweight", "BMI_Category_Normal", "BMI_Category_Obese", "Sum_of_Two_Features"]  # replace "Sum_of_Two_Features" with actual feature name if any

        # Filter features and shap values
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
            values=filtered_shap_vals,
            base_values=base_values,
            data=input_scaled[:, [features.index(f) for f in filtered_features]],
            feature_names=filtered_features,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(expl, max_display=10, ax=ax)
        st.pyplot(fig)

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
