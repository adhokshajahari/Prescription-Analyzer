import streamlit as st
import json
import base64
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediSight · Prescription Analyzer",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load CSS ────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "uploaded_file_type" not in st.session_state:
    st.session_state.uploaded_file_type = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Groq API — Vision + Text ─────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # vision + text
CHAT_MODEL   = "llama-3.3-70b-versatile"                     # fast text chat


def get_groq_key():
    import os
    key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    return key


def call_groq(prompt, image_data=None, image_mime=None, system_prompt="", max_tokens=4096):
    """Single-turn Groq call. Uses vision model when image provided, else chat model."""
    import requests
    api_key = get_groq_key()
    if not api_key:
        return None, "❌ GROQ_API_KEY not set. Add it in .streamlit/secrets.toml"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build user content
    if image_data and image_mime:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{image_data}"
                }
            },
            {"type": "text", "text": prompt}
        ]
        model = VISION_MODEL
    else:
        content = prompt
        model = VISION_MODEL  # also handles plain text fine

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return None, f"API Error {resp.status_code}: {resp.text}"
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text, None
    except Exception as e:
        return None, str(e)


def call_groq_chat(messages, system_prompt="", max_tokens=1024):
    """Multi-turn chat with Groq."""
    import requests
    api_key = get_groq_key()
    if not api_key:
        return None, "❌ GROQ_API_KEY not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": CHAT_MODEL,
        "messages": api_messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return None, f"API Error {resp.status_code}: {resp.text}"
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text, None
    except Exception as e:
        return None, str(e)


def analyze_prescription(raw_text=None, image_data=None, image_mime=None, pdf_data=None):
    """Send prescription to Groq for structured analysis."""
    system_prompt = """You are MediSight, an expert clinical AI assistant specializing in prescription analysis.
Analyze the provided prescription and return a JSON object ONLY.
Do NOT include markdown fences, backticks, or any extra text — just raw JSON.

Return this exact structure:
{
  "patient": {
    "name": "...",
    "dob": "...",
    "mrn": "..."
  },
  "prescriber": {
    "name": "...",
    "specialty": "..."
  },
  "medications": [
    {
      "name": "...",
      "generic_name": "...",
      "dosage": "...",
      "route": "...",
      "frequency": "...",
      "duration": "...",
      "drug_class": "...",
      "indication": "..."
    }
  ],
  "interactions": [
    {
      "drugs": ["Drug A", "Drug B"],
      "severity": "Minor|Moderate|Major|Critical",
      "description": "...",
      "recommendation": "..."
    }
  ],
  "probable_diagnoses": [
    {
      "condition": "...",
      "confidence": "Low|Moderate|High",
      "supporting_evidence": "..."
    }
  ],
  "safety_flags": ["..."],
  "summary": "..."
}

If a field is unknown, use null. Return ONLY the JSON object, nothing else."""

    if pdf_data:
        return None, (
            "⚠️ Groq API does not support direct PDF uploads. "
            "Please copy-paste the prescription text into the text tab, "
            "or upload an image (PNG/JPG/WEBP) of the prescription."
        )

    if raw_text:
        prompt = f"Prescription text to analyze:\n\n{raw_text}"
    else:
        prompt = "Analyze the prescription in the provided image thoroughly."

    result, error = call_groq(
        prompt=prompt,
        image_data=image_data,
        image_mime=image_mime,
        system_prompt=system_prompt,
        max_tokens=4096,
    )

    if error:
        return None, error

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        return None, f"Failed to parse response:\n{result[:500]}"


# ─── UI Helpers ──────────────────────────────────────────────────────────────
def get_severity_color(severity):
    return {"Critical": "#ff4444", "Major": "#ff8800",
            "Moderate": "#ffcc00", "Minor": "#44bb88"}.get(severity, "#888888")


def severity_badge(severity):
    color = get_severity_color(severity)
    bg_map = {
        "Critical": "rgba(255,68,68,0.12)", "Major": "rgba(255,136,0,0.12)",
        "Moderate": "rgba(255,204,0,0.12)", "Minor": "rgba(68,187,136,0.12)",
    }
    bg = bg_map.get(severity, "rgba(136,136,136,0.12)")
    return (f'<span style="background:{bg};color:{color};padding:2px 10px;'
            f'border-radius:20px;font-size:0.75rem;font-weight:600;'
            f'letter-spacing:0.05em;border:1px solid {color}30">'
            f'{severity.upper()}</span>')


def confidence_badge(confidence):
    color = {"High": "#44bb88", "Moderate": "#ffcc00", "Low": "#ff8800"}.get(confidence, "#888")
    return f'<span style="color:{color};font-size:0.78rem;font-weight:600">● {confidence}</span>'


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-logo">⚕ MediSight<span class="hero-ai"> AI</span></div>
  <div class="hero-sub">Prescription Intelligence Platform · Powered by Groq</div>
  <div class="hero-tagline">Intelligence Meets Clinical Precision</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
  ⚠️ <strong>Medical Disclaimer:</strong> MediSight is an AI-assisted tool for clinical support only.
  All outputs must be reviewed by a qualified healthcare professional. Not a substitute for medical judgment.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
col_input, col_results = st.columns([1, 1.6], gap="large")

# ─── LEFT: Input Panel ───────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="panel-title">📋 Prescription Input</div>', unsafe_allow_html=True)

    tab_file, tab_text = st.tabs(["📎 Upload Image", "✏️ Paste Text"])

    with tab_file:
        st.info("📌 Supports PNG, JPG, WEBP prescription scans. For PDFs, use the Paste Text tab.")
        uploaded = st.file_uploader(
            "Upload Prescription Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a clear scan or photo of the prescription",
            label_visibility="collapsed",
        )
        if uploaded:
            file_bytes = uploaded.read()
            b64 = base64.standard_b64encode(file_bytes).decode()
            ftype = uploaded.type
            if ftype.startswith("image/"):
                st.session_state.uploaded_file_data = b64
                st.session_state.uploaded_file_type = ftype
                st.image(file_bytes, caption="Uploaded Prescription", use_container_width=True)
            else:
                st.error("Unsupported file type.")

    with tab_text:
        if st.button("Load Sample Prescription", use_container_width=True, key="sample_btn"):
            st.session_state.raw_text = """PRESCRIPTION

Patient: John Michael Doe
Date of Birth: 15/03/1965
MRN: 7823901

Prescriber: Dr. Sarah Chen, MD — Cardiology
Date: 2025-09-10

1. Metformin HCl 500mg — Oral — Twice daily with meals — 90 days
2. Lisinopril 10mg — Oral — Once daily — 90 days
3. Atorvastatin 40mg — Oral — Once daily at bedtime — 90 days
4. Aspirin 81mg — Oral — Once daily — Indefinite
5. Warfarin 5mg — Oral — Once daily — 90 days (INR monitoring required)
6. Ibuprofen 400mg — Oral — As needed for pain — 30 days

Pharmacy: MedPlus Pharmacy, Andheri West, Mumbai"""

        text_input = st.text_area(
            "Paste prescription text",
            value=st.session_state.raw_text,
            height=280,
            placeholder="Paste prescription text here...\n\nInclude: patient info, medications, dosages, frequency, prescriber details.",
            label_visibility="collapsed",
        )
        st.session_state.raw_text = text_input

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔬 Analyze Prescription", use_container_width=True, type="primary", key="analyze_btn"):
        image_data = image_mime = pdf_data = raw_text = None

        if st.session_state.uploaded_file_data:
            ftype = st.session_state.uploaded_file_type
            if ftype == "pdf":
                pdf_data = st.session_state.uploaded_file_data
            else:
                image_data = st.session_state.uploaded_file_data
                image_mime = ftype
        elif st.session_state.raw_text.strip():
            raw_text = st.session_state.raw_text
        else:
            st.warning("⚠️ Please upload an image or paste prescription text first.")
            st.stop()

        with st.spinner("🧬 Analyzing prescription with Groq AI..."):
            result, error = analyze_prescription(
                raw_text=raw_text,
                image_data=image_data,
                image_mime=image_mime,
                pdf_data=pdf_data,
            )

        if error:
            st.error(f"Analysis failed: {error}")
        else:
            st.session_state.analysis_result = result
            st.session_state.chat_history = []
            st.success("✅ Analysis complete!")
            st.rerun()

    if st.session_state.analysis_result:
        r = st.session_state.analysis_result
        meds = r.get("medications", [])
        ints = r.get("interactions", [])
        critical = [i for i in ints if i.get("severity") in ("Critical", "Major")]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Quick Stats</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-card"><div class="stat-num">{len(meds)}</div><div class="stat-label">Medications</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card warn"><div class="stat-num">{len(ints)}</div><div class="stat-label">Interactions</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card {"danger" if critical else ""}"><div class="stat-num">{len(critical)}</div><div class="stat-label">High Alerts</div></div>', unsafe_allow_html=True)


# ─── RIGHT: Results Panel ─────────────────────────────────────────────────────
with col_results:
    if not st.session_state.analysis_result:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">⚕</div>
          <div class="empty-title">Ready to Analyze</div>
          <div class="empty-sub">Upload a prescription image or paste text<br>to see AI-powered drug interaction analysis,<br>dosage verification, and clinical insights.</div>
          <div class="feature-pills">
            <span class="pill">💊 Drug Interactions</span>
            <span class="pill">🧪 Dosage Check</span>
            <span class="pill">🔬 Diagnosis Hints</span>
            <span class="pill">⚠️ Safety Flags</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.analysis_result
        t1, t2, t3, t4, t5 = st.tabs([
            "💊 Medications", "⚠️ Interactions", "🔬 Diagnosis", "👤 Patient", "💬 Ask AI"
        ])

        # ── TAB 1: Medications ───────────────────────────────────────────
        with t1:
            meds = r.get("medications", [])
            if not meds:
                st.info("No medications extracted.")
            else:
                classes = {}
                for m in meds:
                    dc = m.get("drug_class") or "Other"
                    classes[dc] = classes.get(dc, 0) + 1
                pills_html = " ".join([
                    f'<span class="class-pill">{k} <strong>x{v}</strong></span>'
                    for k, v in classes.items()
                ])
                st.markdown(f'<div class="class-summary">{pills_html}</div>', unsafe_allow_html=True)
                for i, med in enumerate(meds):
                    with st.expander(f"💊 {med.get('name', 'Unknown')} — {med.get('dosage', '')}", expanded=i == 0):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Generic:** {med.get('generic_name') or '—'}")
                            st.markdown(f"**Route:** {med.get('route') or '—'}")
                            st.markdown(f"**Frequency:** {med.get('frequency') or '—'}")
                        with col_b:
                            st.markdown(f"**Duration:** {med.get('duration') or '—'}")
                            st.markdown(f"**Class:** {med.get('drug_class') or '—'}")
                            st.markdown(f"**Indication:** {med.get('indication') or '—'}")

        # ── TAB 2: Interactions ──────────────────────────────────────────
        with t2:
            ints = r.get("interactions", [])
            flags = r.get("safety_flags", [])
            if flags:
                st.markdown('<div class="section-label">🚨 Safety Flags</div>', unsafe_allow_html=True)
                for flag in flags:
                    st.markdown(f'<div class="safety-flag">⚠️ {flag}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            if not ints:
                st.markdown("""
                <div class="no-interactions">
                  <div style="font-size:2rem">✅</div>
                  <div>No significant drug interactions detected</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sev_order = {"Critical": 0, "Major": 1, "Moderate": 2, "Minor": 3}
                for interaction in sorted(ints, key=lambda x: sev_order.get(x.get("severity", "Minor"), 3)):
                    sev = interaction.get("severity", "Minor")
                    drugs = interaction.get("drugs", [])
                    color = get_severity_color(sev)
                    st.markdown(f"""
                    <div class="interaction-card" style="border-left:3px solid {color}">
                      <div class="interaction-header">
                        <span class="interaction-drugs">{"  ↔  ".join(drugs)}</span>
                        {severity_badge(sev)}
                      </div>
                      <div class="interaction-desc">{interaction.get('description', '')}</div>
                      <div class="interaction-rec">💡 {interaction.get('recommendation', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── TAB 3: Diagnosis ──────────────────────────────────────────────
        with t3:
            st.markdown("""
            <div class="ai-disclaimer">
              🤖 <strong>AI Suggestion Only</strong> — These are probability-based suggestions derived from
              medication patterns. They are NOT a clinical diagnosis. Always consult a qualified physician.
            </div>
            """, unsafe_allow_html=True)
            diags = r.get("probable_diagnoses", [])
            if not diags:
                st.info("No diagnosis suggestions available.")
            else:
                for d in diags:
                    st.markdown(f"""
                    <div class="diagnosis-card">
                      <div class="diag-header">
                        <span class="diag-name">{d.get('condition', '')}</span>
                        {confidence_badge(d.get('confidence', 'Low'))}
                      </div>
                      <div class="diag-evidence">📎 {d.get('supporting_evidence', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            summary = r.get("summary")
            if summary:
                st.markdown('<div class="section-label" style="margin-top:1.5rem">📋 Clinical Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True)

        # ── TAB 4: Patient ────────────────────────────────────────────────
        with t4:
            patient = r.get("patient", {})
            prescriber = r.get("prescriber", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-label">👤 Patient Information</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-row"><span class="info-label">Name</span><span class="info-val">{patient.get('name') or '—'}</span></div>
                  <div class="info-row"><span class="info-label">Date of Birth</span><span class="info-val">{patient.get('dob') or '—'}</span></div>
                  <div class="info-row"><span class="info-label">MRN</span><span class="info-val">{patient.get('mrn') or '—'}</span></div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="section-label">🩺 Prescriber</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-row"><span class="info-label">Name</span><span class="info-val">{prescriber.get('name') or '—'}</span></div>
                  <div class="info-row"><span class="info-label">Specialty</span><span class="info-val">{prescriber.get('specialty') or '—'}</span></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Export Full Analysis (JSON)",
                data=json.dumps(r, indent=2),
                file_name="medisight_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

        # ── TAB 5: Chat ───────────────────────────────────────────────────
        with t5:
            st.markdown('<div class="section-label">💬 Ask about this Prescription</div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
                role_icon = "👤" if msg["role"] == "user" else "⚕"
                st.markdown(f"""
                <div class="{role_class}">
                  <span class="chat-icon">{role_icon}</span>
                  <span class="chat-text">{msg["content"]}</span>
                </div>
                """, unsafe_allow_html=True)

            quick_prompts = [
                "Are there any dangerous combinations here?",
                "What conditions might this patient have?",
                "Are these dosages within normal range?",
                "Summarize this prescription in simple terms",
            ]
            cols = st.columns(2)
            for idx, qp in enumerate(quick_prompts):
                if cols[idx % 2].button(qp, key=f"qp_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": qp})
                    with st.spinner("Thinking..."):
                        context = f"Prescription analysis:\n{json.dumps(r, indent=2)}\n\nQuestion: {qp}"
                        reply, err = call_groq(prompt=context, max_tokens=1024)
                    if reply:
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

            user_q = st.chat_input("Ask anything about this prescription...")
            if user_q:
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.spinner("Thinking..."):
                    system = ("You are MediSight, a clinical AI assistant. Answer questions about "
                              "prescriptions clearly and always remind the user this is AI assistance "
                              "only, not a substitute for professional medical advice.")
                    messages = [{"role": "user", "content": f"Prescription analysis data:\n{json.dumps(r, indent=2)}"}]
                    messages.append({"role": "assistant", "content": "I have reviewed the prescription analysis. How can I help?"})
                    for h in st.session_state.chat_history:
                        messages.append({"role": h["role"], "content": h["content"]})
                    reply, err = call_groq_chat(messages, system_prompt=system, max_tokens=1024)
                if reply:
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>⚕ MediSight AI · Prescription Intelligence Platform</span>
  <span>·</span>
  <span>For clinical support only · Not a substitute for medical advice</span>
  <span>·</span>
  <span>Powered by Groq · Llama 4 Scout</span>
</div>
""", unsafe_allow_html=True)
