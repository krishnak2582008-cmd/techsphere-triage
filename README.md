# AI Triage Queue

Clinical decision-support tool that scores patient cases for disease risk, ranks them by urgency, and explains each flag, so healthcare professionals can prioritize cases for further assessment.

Built for TECHSPHERE WEB-A-THON (Computer Society of India, Jain University).

## Problem Statement

Develop a decision-support tool that analyzes approved medical images or clinical datasets to identify potential signs of disease and help healthcare professionals prioritize cases for further assessment.

## Solution Overview

1. Loads clinical measurements from a public, de-identified dataset (or an uploaded CSV).
2. A Random Forest model estimates each case's likelihood of a suspicious finding.
3. Cases are ranked into a triage queue: High (70% and above), Medium (40-70%), Low (below 40%).
4. Each case shows its top contributing factors (explainability).
5. An LLM (Gemini) writes a short clinician-facing summary, with a safe template fallback if the API is unavailable.

Decision-support only. This tool does not diagnose. All flags require review by a qualified healthcare professional.

## Tech Stack

Python, Streamlit, scikit-learn, pandas, Google Gemini API

## Architecture

Clinical data -> Random Forest risk score -> Triage tiers -> Ranked queue -> Top contributing factors -> LLM clinical summary (with fallback)

## Dataset

Breast Cancer Wisconsin (Diagnostic), public and de-identified, from scikit-learn. Queue cases come from a held-out test set the model never trained on.

## Setup

git clone YOUR_REPO_URL
cd techsphere-triage
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Optional: copy .env.example to .env and add your GEMINI_API_KEY.

## Live Demo

LIVE_URL_HERE

## Screenshots

Add screenshots here.

## Limitations

Demo model; not clinically validated; not for real patient care.
