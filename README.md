# ⚕ MediSight — Prescription Analyzer

> AI-powered prescription analysis, drug interaction detection, and clinical insights.  
> Built with Streamlit + Claude (Anthropic API).

---

## 🖥️ Run Locally

### Step 1 — Clone / Download the project

```bash
git clone https://github.com/YOUR_USERNAME/medisight.git
cd medisight
```

Or just copy the files into a folder named `medisight/`.

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate:
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your Anthropic API key

Edit `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

Get your key at → https://console.anthropic.com/

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial MediSight app"
git remote add origin https://github.com/YOUR_USERNAME/medisight.git
git push -u origin main
```

> ⚠️ Do NOT commit `.streamlit/secrets.toml` with your real API key.  
> Add it to `.gitignore` first:
> ```
> echo ".streamlit/secrets.toml" >> .gitignore
> ```

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to → https://share.streamlit.io
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **"Advanced settings"** → **Secrets** tab
6. Paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
7. Click **Deploy!**

Your app will be live at:  
`https://YOUR_USERNAME-medisight-app-XXXX.streamlit.app`

---

## 📁 Project Structure

```
medisight/
├── app.py                  ← Main Streamlit application
├── style.css               ← Dark pharma-AI theme (PharmaZep-inspired)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   ├── config.toml         ← Streamlit theme config
│   └── secrets.toml        ← API keys (DO NOT commit)
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📎 PDF Upload | Parse PDF prescriptions via Claude's document API |
| 🖼️ Image Upload | OCR-style analysis of prescription scans (JPG/PNG/WebP) |
| ✏️ Text Input | Paste raw prescription text |
| 💊 Medication Extraction | Name, dosage, route, frequency, duration, drug class |
| ⚠️ Drug Interactions | Severity-rated (Minor → Critical), with recommendations |
| 🔬 Diagnosis Hints | AI-suggested probable diagnoses with confidence levels |
| 🚨 Safety Flags | Automatic detection of high-risk combinations |
| 💬 Ask AI | Chat interface to ask questions about the prescription |
| ⬇️ Export | Download full analysis as JSON |

---

## 🔐 Security Notes

- API keys are stored in Streamlit secrets — never in code
- No patient data is stored — all analysis is stateless per session
- Always add a medical disclaimer for production use
- For HIPAA compliance, use a private cloud deployment with encryption at rest

---

## 🛠️ Customization Ideas

- Add a real drug database (RxNorm API or DrugBank) for verified interactions
- Connect a physician locator via Google Maps API
- Add PDF report generation with `reportlab`
- Add authentication with `streamlit-authenticator`
- Add a patient history dashboard with a database backend

---

## ⚕ Disclaimer

MediSight is an AI-assisted clinical support tool. All outputs must be reviewed by a qualified healthcare professional. It is NOT a substitute for professional medical judgment, diagnosis, or treatment.
