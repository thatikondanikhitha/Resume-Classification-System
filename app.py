import streamlit as st
import joblib
import pdfplumber
import docx

# ---------- Page Config ----------
st.set_page_config(
    page_title="Resume Classification",
    page_icon="📄"
)

# ---------- Light Blue Themed Background (SAFE) ----------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1526378722445-4c4b6d0b2c90");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = joblib.load("svm_resume_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

svm_model, tfidf = load_model()

# ---------- Text Extraction ----------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_text_from_docx(file):
    document = docx.Document(file)
    return "\n".join([p.text for p in document.paragraphs])

# ---------- App UI ----------
st.title("📄 Resume Classification System")
st.write("Upload a resume or paste resume text to classify it.")

# ---- File Upload (NO WHITE BAR) ----
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"],
    label_visibility="collapsed"
)

# ---- Text Input (NO WHITE BAR) ----
resume_text_input = st.text_area(
    "Paste Resume Text",
    height=150,
    placeholder="Paste resume text here...",
    label_visibility="collapsed"
)

# ---------- Prediction ----------
resume_text = ""

if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

elif resume_text_input.strip():
    resume_text = resume_text_input

if resume_text.strip():
    vector = tfidf.transform([resume_text])
    prediction = svm_model.predict(vector)[0]
    st.success(f"✅ Predicted Category: **{prediction}**")
