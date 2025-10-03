import os
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import pandas as pd
import pdfplumber   # <-- use pdfplumber instead of PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""   # safe: handle empty pages
    return text

# Match resumes with job description
def match_resumes(resume_texts, job_description):
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    scores = []
    for filename, text in resume_texts.items():
        resume_embedding = model.encode(text, convert_to_tensor=True)
        similarity = util.cos_sim(jd_embedding, resume_embedding).item()
        scores.append({"Resume": filename, "Score": round(similarity * 100, 2)})
    return sorted(scores, key=lambda x: x["Score"], reverse=True)

# ---------------- Streamlit App ----------------
st.title("Resume shortlist")

# Job description input
job_description = st.text_area("Paste the Job Description:")

# Resume upload (multiple PDFs)
uploaded_files = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True)

# Threshold slider
threshold = st.slider("Minimum Match % to include in results", 0, 100, 30)

if st.button("Process Resumes") and job_description and uploaded_files:
    st.info("Extracting and matching resumes... please wait!")

    resume_texts = {}
    for file in uploaded_files:
        path = os.path.join("resumes", file.name)
        os.makedirs("resumes", exist_ok=True)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        resume_texts[file.name] = extract_text_from_pdf(path)

    # Match resumes
    results = match_resumes(resume_texts, job_description)
    df = pd.DataFrame(results)

    # Apply threshold filter
    filtered_df = df[df["Score"] >= threshold]

    if filtered_df.empty:
        st.warning("No resumes matched the job description closely enough.")
    else:
        st.success("Matching complete!")
        st.dataframe(filtered_df)

        # Export option
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Report (CSV)", csv, "resume_report.csv", "text/csv")
