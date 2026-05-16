import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import io
import os
from datetime import datetime
from fpdf import FPDF

# -- Page config -------------------------------------------------------
st.set_page_config(
    page_title="EduPredict - Student Performance AI",
    page_icon="https://img.icons8.com/emoji/48/graduation-cap-emoji.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS --------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background: #f7f5f0; }

  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    line-height: 1.15;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: #6b7280;
    font-weight: 300;
    margin-top: 0.3rem;
  }
  .badge-pass {
    display: inline-block;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white;
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    padding: 0.5rem 1.6rem;
    border-radius: 50px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(34,197,94,0.35);
  }
  .badge-fail {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    padding: 0.5rem 1.6rem;
    border-radius: 50px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(239,68,68,0.35);
  }
  .metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 4px solid #6366f1;
    margin-bottom: 1rem;
  }
  .metric-label { font-size: 0.78rem; color: #9ca3af; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.7rem; font-weight: 600; color: #1a1a2e; margin-top: 0.1rem; }
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #1a1a2e;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.4rem;
  }
  .suggestion-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    border-left: 3px solid;
  }
  .stSidebar { background: #1a1a2e !important; }
  .stSidebar * { color: #e5e7eb !important; }
  .stSidebar .stSelectbox label, .stSidebar .stSlider label { color: #9ca3af !important; font-size: 0.82rem !important; }
  [data-testid="stSidebar"] { background: #1a1a2e; }
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
  div[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
  }
</style>
""", unsafe_allow_html=True)

# -- Load model --------------------------------------------------------
MODEL_PATH = r'E:\Student_Performance\artifacts\models\best_model.pkl'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
FEATURES = list(model.feature_names_in_)

# -- Model stats (pre-computed) ----------------------------------------
MODEL_STATS = {
    "Accuracy":     0.8861,
    "F1 Score":     0.9109,
    "Precision":    0.9583,
    "Recall":       0.8679,
    "Test Samples": 79,
    "Algorithm":    "XGBoost",
}

# -- Helper: build input dataframe ------------------------------------
def build_input(study, absences, g1, g2, internet, famsup, failures,
                sex, age, address, medu, fedu, reason, guardian,
                schoolsup, paid, higher, romantic, goout, dalc, health):
    row = {f: 0 for f in FEATURES}
    row.update({
        'studytime': study,
        'absences':  absences,
        'G1':        g1,
        'G2':        g2,
        'internet':  1 if internet == "Yes" else 0,
        'famsup':    1 if famsup   == "Yes" else 0,
        'failures':  failures,
        'sex':       1 if sex      == "Male" else 0,
        'age':       age,
        'address':   1 if address  == "Urban" else 0,
        'Medu':      medu,
        'Fedu':      fedu,
        'reason':    {"Course": 0, "Home": 1, "Reputation": 2, "Other": 3}[reason],
        'guardian':  {"Mother": 0, "Father": 1, "Other": 2}[guardian],
        'schoolsup': 1 if schoolsup == "Yes" else 0,
        'paid':      1 if paid      == "Yes" else 0,
        'higher':    1 if higher    == "Yes" else 0,
        'romantic':  1 if romantic  == "Yes" else 0,
        'goout':     goout,
        'Dalc':      dalc,
        'health':    health,
    })
    df = pd.DataFrame([row])[FEATURES]
    return df

# -- Helper: estimated G3 score ---------------------------------------
def estimate_g3_score(g1, g2, prob_pass):
    base    = (g1 + g2) / 2
    blended = base * 0.7 + prob_pass * 20 * 0.3
    return round(min(max(blended, 0), 20), 1)

# -- Helper: performance category -------------------------------------
def perf_category(score):
    if score >= 18:   return ("Excellent",        "#16a34a")
    elif score >= 15: return ("Good",             "#2563eb")
    elif score >= 12: return ("Satisfactory",     "#d97706")
    elif score >= 10: return ("Borderline Pass",  "#f59e0b")
    else:             return ("At Risk",           "#dc2626")

# -- Helper: improvement suggestions ----------------------------------
# NOTE: All text here uses only ASCII / latin-1 safe characters.
#       En-dashes replaced with hyphens; no emoji.
def get_suggestions(study, absences, g1, g2, internet, famsup, failures):
    tips      = []
    avg_grade = (g1 + g2) / 2

    if study <= 1:
        tips.append(("Increase weekly study hours to at least 5-10h", "#6366f1"))
    elif study == 2:
        tips.append(("Push study time above 5h/week for better retention", "#8b5cf6"))

    if absences > 10:
        tips.append((f"High absences ({absences}) are hurting your grade - aim for fewer than 5", "#dc2626"))
    elif absences > 5:
        tips.append(("Reduce absences; attendance strongly correlates with performance", "#f59e0b"))

    if avg_grade < 10:
        tips.append(("Seek extra tutoring or past-paper practice - grades need urgent attention", "#ef4444"))
    elif avg_grade < 13:
        tips.append(("Review weaker subjects; practice active recall instead of re-reading", "#d97706"))

    if failures > 0:
        tips.append((f"Address {failures} prior failure(s) - speak with your teacher for support", "#dc2626"))

    if internet == "No":
        tips.append(("Gain internet access for study resources like Khan Academy and YouTube", "#0284c7"))

    if famsup == "No":
        tips.append(("Join a study group or seek peer support to compensate for low family support", "#7c3aed"))

    if not tips:
        tips.append(("You are on the right track - maintain consistency and aim higher!", "#16a34a"))
        tips.append(("Challenge yourself with harder past papers to stretch your grade further", "#2563eb"))

    return tips

# -- Helper: performance bar chart ------------------------------------
def make_performance_chart(study, absences, g1, g2, failures):
    cats = ['Study\nTime', 'Attendance', 'Grade\nG1', 'Grade\nG2', 'Past\nRecord']
    vals = [
        (study / 4) * 100,
        max(0, (1 - absences / 30)) * 100,
        (g1 / 20) * 100,
        (g2 / 20) * 100,
        max(0, (1 - failures / 3)) * 100,
    ]
    colors = ['#6366f1', '#22c55e', '#f59e0b', '#f59e0b',
              '#ef4444' if failures > 0 else '#22c55e']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#f7f5f0')
    ax.set_facecolor('#f7f5f0')
    bars = ax.barh(cats, vals, color=colors, height=0.55,
                   edgecolor='white', linewidth=1.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (normalized %)", color='#6b7280', fontsize=9)
    ax.tick_params(colors='#374151', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, color='#d1d5db')
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}%', va='center', ha='left', fontsize=9,
                fontweight='600', color='#374151')
    ax.set_title("Student Performance Profile", fontsize=11,
                 fontweight='bold', color='#1a1a2e', pad=10)
    plt.tight_layout()
    return fig

# -- Helper: model metrics chart --------------------------------------
def make_model_chart():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    vals    = [MODEL_STATS['Accuracy'], MODEL_STATS['Precision'],
               MODEL_STATS['Recall'],  MODEL_STATS['F1 Score']]
    colors  = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd']

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    bars = ax.bar(metrics, vals, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color='#6b7280', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, color='#e5e7eb')
    ax.set_axisbelow(True)
    ax.tick_params(colors='#374151', labelsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.1%}', ha='center', va='bottom',
                fontsize=9, fontweight='600', color='#374151')
    ax.set_title("Model Performance (XGBoost - Test Set)", fontsize=10,
                 fontweight='bold', color='#1a1a2e', pad=8)
    plt.tight_layout()
    return fig

# -- PDF Generator ----------------------------------------------------
# IMPORTANT: Every string written to FPDF must be latin-1 safe.
#            No en-dashes (U+2013), em-dashes, curly quotes, or emoji.
def _safe(text):
    """Replace common non-latin-1 chars; strip anything still outside range."""
    replacements = {
        '\u2013': '-',   # en-dash
        '\u2014': '--',  # em-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_pdf(inputs, prediction, prob, g3_est, category, suggestions):
    pdf = FPDF()
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(190, 10, _safe("Student Performance Report"), align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 180, 200)
    pdf.set_xy(10, 22)
    pdf.cell(190, 8,
             _safe(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}"),
             align="C")
    pdf.ln(20)

    # Result line
    pdf.set_font("Helvetica", "B", 14)
    color = (22, 163, 74) if prediction == 1 else (220, 38, 38)
    pdf.set_text_color(*color)
    result_text = "PREDICTED: PASS" if prediction == 1 else "PREDICTED: FAIL"
    pdf.cell(0, 12, _safe(result_text), ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8,
             _safe(f"Estimated G3 Score: {g3_est:.2f}/20     "
                   f"Category: {category[0]}     "
                   f"Confidence: {max(prob):.1%}"),
             ln=True)
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(200, 200, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Student input summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 8, _safe("Student Input Summary"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)

    input_pairs = [
        ("Study Time",      f"{inputs['study']} (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)"),
        ("Absences",        str(inputs['absences'])),
        ("Grade G1",        f"{inputs['g1']} / 20"),
        ("Grade G2",        f"{inputs['g2']} / 20"),
        ("Internet Access", inputs['internet']),
        ("Family Support",  inputs['famsup']),
        ("Past Failures",   str(inputs['failures'])),
        ("Sex",             inputs['sex']),
        ("Age",             str(inputs['age'])),
    ]

    for k, v in input_pairs:
        pdf.cell(0, 6, _safe(f"  {k}: {v}"), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Improvement suggestions
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 8, _safe("Personalized Improvement Suggestions"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    for (tip, _) in suggestions:
        pdf.cell(0, 7, _safe(f"  * {tip}"), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Model performance summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 8, _safe("Model Performance Summary"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 80)
    pdf.cell(0, 6, _safe(f"  Algorithm   : {MODEL_STATS['Algorithm']}"),   ln=True)
    pdf.cell(0, 6, _safe(f"  Accuracy    : {MODEL_STATS['Accuracy']:.1%}"), ln=True)
    pdf.cell(0, 6, _safe(f"  F1 Score    : {MODEL_STATS['F1 Score']:.1%}"), ln=True)
    pdf.cell(0, 6, _safe(f"  Precision   : {MODEL_STATS['Precision']:.1%}"),ln=True)
    pdf.cell(0, 6, _safe(f"  Recall      : {MODEL_STATS['Recall']:.1%}"),   ln=True)
    pdf.cell(0, 6, _safe(f"  Test Samples: {MODEL_STATS['Test Samples']}"), ln=True)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 180)
    pdf.cell(0, 5,
             _safe("EduPredict - AI-powered Student Performance Predictor - For educational use only"),
             align="C")

    return pdf.output(dest='S').encode('latin-1') if isinstance(pdf.output(dest='S'), str) else pdf.output(dest='S')


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <span style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#e5e7eb'>EduPredict</span>
      <div style='font-size:0.8rem;color:#9ca3af;margin-top:0.2rem'>Student Performance AI</div>
    </div>
    <hr style='border-color:#374151;margin:0.5rem 0 1rem'/>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.78rem;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:.07em;margin-bottom:.5rem'>Academic Profile</div>",
                unsafe_allow_html=True)

    # NOTE: format_func strings use hyphens, NOT en-dashes
    study = st.selectbox(
        "Study Hours / Week",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "<2 hours",
            2: "2-5 hours",
            3: "5-10 hours",
            4: ">10 hours"
        }[x],
        index=1
    )
    absences = st.slider("School Absences", 0, 30, 4)
    g1       = st.slider("First Period Grade (G1)", 0, 20, 10)
    g2       = st.slider("Second Period Grade (G2)", 0, 20, 11)
    failures = st.selectbox("Past Course Failures", [0, 1, 2, 3], index=0)

    st.markdown("<hr style='border-color:#374151;margin:0.8rem 0'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:.07em;margin-bottom:.5rem'>Personal & Home</div>",
                unsafe_allow_html=True)

    internet = st.selectbox("Internet at Home", ["Yes", "No"])
    famsup   = st.selectbox("Family Educational Support", ["Yes", "No"])
    sex      = st.selectbox("Sex", ["Female", "Male"])
    age      = st.slider("Age", 15, 22, 17)
    address  = st.selectbox("Address Type", ["Urban", "Rural"])

    st.markdown("<hr style='border-color:#374151;margin:0.8rem 0'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:.07em;margin-bottom:.5rem'>Additional Info</div>",
                unsafe_allow_html=True)

    medu      = st.slider("Mother's Education (0-4)", 0, 4, 2)
    fedu      = st.slider("Father's Education (0-4)", 0, 4, 2)
    reason    = st.selectbox("School Choice Reason", ["Course", "Home", "Reputation", "Other"])
    guardian  = st.selectbox("Guardian", ["Mother", "Father", "Other"])
    schoolsup = st.selectbox("School Extra Support", ["Yes", "No"])
    paid      = st.selectbox("Paid Extra Classes", ["No", "Yes"])
    higher    = st.selectbox("Wants Higher Education", ["Yes", "No"])
    romantic  = st.selectbox("In Romantic Relationship", ["No", "Yes"])
    goout     = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
    dalc      = st.slider("Workday Alcohol Use (1-5)", 1, 5, 1)
    health    = st.slider("Health Status (1-5)", 1, 5, 3)

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Performance", use_container_width=True)


# =====================================================================
# MAIN AREA
# =====================================================================
st.markdown("""
<div class='hero-title'>Student Performance Predictor</div>
<div class='hero-sub'>AI-powered analysis using XGBoost trained on real student data</div>
<br/>
""", unsafe_allow_html=True)

if not predict_btn:
    # Landing state
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='metric-card'>
          <div class='metric-label'>Model Algorithm</div>
          <div class='metric-value'>XGBoost</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
          <div class='metric-label'>Test Accuracy</div>
          <div class='metric-value'>88.6%</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
          <div class='metric-label'>F1 Score</div>
          <div class='metric-value'>91.1%</div></div>""", unsafe_allow_html=True)

    st.info("Configure student inputs in the sidebar, then click 'Predict Performance'")

    st.markdown("<div class='section-title'>Model Performance Summary</div>",
                unsafe_allow_html=True)
    fig_m = make_model_chart()
    st.pyplot(fig_m, use_container_width=False)

else:
    # Build input & predict
    input_df = build_input(study, absences, g1, g2, internet, famsup,
                           failures, sex, age, address, medu, fedu,
                           reason, guardian, schoolsup, paid, higher,
                           romantic, goout, dalc, health)

    prob       = model.predict_proba(input_df)[0]
    g3_est     = estimate_g3_score(g1, g2, prob[1])
    prediction = 0 if g3_est <= 10 else 1
    cat, cat_color = perf_category(g3_est)
    suggestions    = get_suggestions(study, absences, g1, g2, internet, famsup, failures)
    inputs_dict    = dict(study=study, absences=absences, g1=g1, g2=g2,
                          internet=internet, famsup=famsup, failures=failures,
                          sex=sex, age=age)

    # Result row
    r1, r2, r3, r4 = st.columns([1.5, 1, 1, 1])

    with r1:
        badge = "badge-pass" if prediction == 1 else "badge-fail"
        label = "PASS" if prediction == 1 else "FAIL"
        st.markdown(
            f"<div style='margin-top:0.3rem'><span class='{badge}'>{label}</span></div>",
            unsafe_allow_html=True
        )
    with r2:
        st.metric("Confidence", f"{max(prob):.1%}")
    with r3:
        st.metric("Est. G3 Score", f"{g3_est:.2f} / 20")
    with r4:
        st.metric("Category", cat)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Chart + suggestions
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("<div class='section-title'>Performance Profile</div>",
                    unsafe_allow_html=True)
        fig_perf = make_performance_chart(study, absences, g1, g2, failures)
        st.pyplot(fig_perf, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Improvement Suggestions</div>",
                    unsafe_allow_html=True)
        for (tip, color) in suggestions:
            st.markdown(
                f"<div class='suggestion-card' style='border-color:{color}'>"
                f"<span style='font-size:0.9rem;color:#374151'>{tip}</span></div>",
                unsafe_allow_html=True
            )

    # Model performance section
    st.markdown("<div class='section-title'>Model Performance Summary</div>",
                unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, val) in zip([m1, m2, m3, m4, m5], [
        ("Algorithm", MODEL_STATS["Algorithm"]),
        ("Accuracy",  f"{MODEL_STATS['Accuracy']:.1%}"),
        ("F1 Score",  f"{MODEL_STATS['F1 Score']:.1%}"),
        ("Precision", f"{MODEL_STATS['Precision']:.1%}"),
        ("Recall",    f"{MODEL_STATS['Recall']:.1%}"),
    ]):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='font-size:1.2rem'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    fig_model = make_model_chart()
    st.pyplot(fig_model, use_container_width=False)

    # PDF export
    st.markdown("<div class='section-title'>Export Report</div>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf(
        inputs_dict, prediction, prob, g3_est,
        (cat, cat_color), suggestions
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"EduPredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=False
    )