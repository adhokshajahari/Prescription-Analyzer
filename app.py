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
# Strategy: try base64 embed via components.html (bypasses st.markdown sanitization)
# Fallback: animated CSS gradient hero if video file not found
import streamlit.components.v1 as _hero_components

_vpath = Path(__file__).parent / "static" / "hero_bg.mp4"
_vsrc = ""
if _vpath.exists():
    with open(_vpath, "rb") as _vf:
        _vsrc = "data:video/mp4;base64," + base64.b64encode(_vf.read()).decode()

if _vsrc:
    _video_html = f"""
    <video id="hv" autoplay muted loop playsinline webkit-playsinline
           preload="auto"
           style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                  min-width:100%;min-height:100%;object-fit:cover;z-index:0;opacity:0.65;">
      <source src="{_vsrc}" type="video/mp4">
    </video>"""
else:
    # Animated gradient fallback — looks great, zero dependencies
    _video_html = ""

_hero_components.html(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:transparent;overflow:hidden}}
.hero{{
  position:relative;width:100%;height:380px;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
  background:linear-gradient(135deg,#0a0a1a 0%,#0d1f2d 30%,#0a0a1a 60%,#141428 100%);
  background-size:400% 400%;
  animation:gradMove 8s ease infinite;
}}
@keyframes gradMove{{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
video.bgvid{{
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  min-width:100%;min-height:100%;
  width:auto;height:auto;
  object-fit:cover;z-index:1;opacity:0.7;
}}
.overlay{{position:absolute;inset:0;z-index:2;background:linear-gradient(to bottom,rgba(0,0,0,0.1) 0%,rgba(0,0,0,0.65) 100%)}}
.content{{position:relative;z-index:3;text-align:center;color:#fff;font-family:Georgia,serif;padding:20px}}
.logo{{font-size:clamp(2rem,6vw,3.6rem);font-style:italic;font-weight:700;
       color:#e8e0d0;text-shadow:0 2px 24px rgba(0,0,0,0.9)}}
.ai{{font-size:0.52em;font-style:normal;font-weight:300;letter-spacing:.18em;vertical-align:super;color:#b0a898}}
.sub{{margin-top:10px;font-size:clamp(0.6rem,1.6vw,0.8rem);letter-spacing:.25em;
      font-family:'Helvetica Neue',sans-serif;font-weight:300;color:#908878;text-transform:uppercase}}
.tag{{margin-top:14px;font-size:clamp(0.85rem,2.2vw,1.2rem);font-style:italic;
      color:#c0b8a8;text-shadow:0 1px 10px rgba(0,0,0,0.7)}}
.star{{position:absolute;border-radius:50%;background:#fff;
       animation:twinkle var(--d,3s) ease-in-out infinite;opacity:0;z-index:1}}
@keyframes twinkle{{0%,100%{{opacity:0}}50%{{opacity:var(--o,.5)}}}}
</style></head><body>
<div class="hero" id="hero">
  {"" if not _vsrc else f'<video class="bgvid" id="hv" autoplay muted loop playsinline webkit-playsinline preload="auto"><source src="{_vsrc}" type="video/mp4"></video>'}
  <div class="star" style="width:2px;height:2px;top:15%;left:20%;--d:2.5s;--o:.7;animation-delay:0s"></div>
  <div class="star" style="width:1px;height:1px;top:30%;left:70%;--d:3.2s;--o:.5;animation-delay:.8s"></div>
  <div class="star" style="width:2px;height:2px;top:55%;left:45%;--d:2.8s;--o:.8;animation-delay:.3s"></div>
  <div class="star" style="width:1px;height:1px;top:20%;left:85%;--d:3.5s;--o:.4;animation-delay:1.2s"></div>
  <div class="star" style="width:2px;height:2px;top:70%;left:15%;--d:2.2s;--o:.9;animation-delay:.5s"></div>
  <div class="star" style="width:1px;height:1px;top:40%;left:55%;--d:3.8s;--o:.6;animation-delay:1.8s"></div>
  <div class="star" style="width:2px;height:2px;top:80%;left:80%;--d:2.6s;--o:.7;animation-delay:.9s"></div>
  <div class="star" style="width:1px;height:1px;top:10%;left:40%;--d:3.1s;--o:.5;animation-delay:.2s"></div>
  <div class="star" style="width:2px;height:2px;top:60%;left:30%;--d:2.9s;--o:.8;animation-delay:1.5s"></div>
  <div class="star" style="width:2px;height:2px;top:25%;left:92%;--d:2.3s;--o:.9;animation-delay:1.1s"></div>
  <div class="overlay"></div>
  <div class="content">
    <div class="logo">⚕ MediSight<span class="ai"> AI</span></div>
    <div class="sub">Prescription Intelligence Platform · Powered by Groq</div>
    <div class="tag">Intelligence Meets Clinical Precision</div>
  </div>
</div>
<script>
(function(){{
  var v=document.getElementById("hv");
  if(!v)return;
  function p(){{
    v.muted=true;
    var r=v.play();
    if(r&&r.catch)r.catch(function(){{}});
  }}
  // Try immediately
  p();
  // Try after load
  window.addEventListener("load",p);
  // iOS/Android: on first interaction
  document.addEventListener("touchstart",p,{{once:true,passive:true}});
  document.addEventListener("click",p,{{once:true}});
  // If video fails entirely, hide it — gradient still shows
  v.addEventListener("error",function(){{v.style.display="none";}});
}})();
</script>
</body></html>""", height=390, scrolling=False)

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

        mobile_col1, mobile_col2 = st.columns(2)

        with mobile_col1:
            st.markdown("**📁 Upload from Gallery**")
            uploaded = st.file_uploader(
                "Upload Prescription Image",
                type=["png", "jpg", "jpeg", "webp"],
                help="Choose an existing image from your device gallery",
                label_visibility="collapsed",
                key="gallery_uploader",
            )

        with mobile_col2:
            st.markdown("**📷 Take a Photo**")
            camera_image = st.camera_input(
                "Take a photo of the prescription",
                label_visibility="collapsed",
                key="camera_input",
            )

        active_file = camera_image or uploaded

        if active_file:
            file_bytes = active_file.read()
            b64 = base64.standard_b64encode(file_bytes).decode()
            if hasattr(active_file, "type") and active_file.type:
                ftype = active_file.type
            else:
                ftype = "image/jpeg"
            if ftype.startswith("image/"):
                st.session_state.uploaded_file_data = b64
                st.session_state.uploaded_file_type = ftype
                st.image(file_bytes, caption="Prescription Preview", use_container_width=True)
            else:
                st.error("Unsupported file type. Please upload PNG, JPG, or WEBP.")

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

# ═══════════════════════════════════════════════════════════════════════════
# 🔊 ELDER-FRIENDLY VOICE & CARE GUIDE — Full width, always visible
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_result:
    import streamlit.components.v1 as components

    r_v = st.session_state.analysis_result
    meds_v  = r_v.get("medications", [])
    diags_v = r_v.get("probable_diagnoses", [])
    summ_v  = r_v.get("summary", "")

    # Init session state
    for k, v in [("voice_text", ""), ("voice_lang_tag", "en-IN"),
                 ("voice_lang_label", "English"), ("elder_simple_text", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Build English medicine schedule
    med_lines = []
    for i, m in enumerate(meds_v, 1):
        name  = m.get("name") or "Unknown medicine"
        dose  = m.get("dosage") or ""
        freq  = m.get("frequency") or ""
        dur   = m.get("duration") or ""
        ind   = m.get("indication") or ""
        parts = [f"Medicine {i}: {name}"]
        if dose: parts.append(f"Dose {dose}")
        if freq: parts.append(f"Take {freq}")
        if dur:  parts.append(f"for {dur}")
        if ind:  parts.append(f"This helps with {ind}")
        med_lines.append(". ".join(parts))

    diag_lines = [d.get("condition","") for d in diags_v if d.get("condition")]
    eng_full = ". ".join(med_lines)
    if diag_lines:
        eng_full += ". Your doctor thinks you may have: " + ", ".join(diag_lines)
    if summ_v:
        eng_full += ". Summary: " + summ_v

    if not st.session_state.voice_text:
        st.session_state.voice_text = eng_full

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION HEADER ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#112244);
                border:2px solid #2255aa;border-radius:20px;
                padding:1.6rem 2rem;margin-bottom:1.5rem;text-align:center;">
      <div style="font-size:2.2rem;font-weight:800;color:#ffffff;letter-spacing:-0.5px;margin-bottom:0.4rem;">
        🔊 बोलकर सुनें · Listen & Understand
      </div>
      <div style="font-size:1.1rem;color:#90b8f0;font-weight:500;">
        Designed for elderly patients &amp; those who prefer listening over reading
        &nbsp;·&nbsp; 12 Indian languages supported
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ROW 1: Language picker + Translate + Simple Explainer ──────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])

    languages = {
        "🇬🇧 English":   ("en-IN", None),
        "🇮🇳 Hindi":     ("hi-IN", "Translate to very simple Hindi (Devanagari). Use short sentences. Keep medicine names in English."),
        "🌿 Kannada":    ("kn-IN", "Translate to very simple Kannada. Short sentences. Keep medicine names in English."),
        "🌺 Tamil":      ("ta-IN", "Translate to very simple Tamil. Short sentences. Keep medicine names in English."),
        "🌸 Telugu":     ("te-IN", "Translate to very simple Telugu. Short sentences. Keep medicine names in English."),
        "🦚 Malayalam":  ("ml-IN", "Translate to very simple Malayalam. Short sentences. Keep medicine names in English."),
        "🌼 Marathi":    ("mr-IN", "Translate to very simple Marathi. Short sentences. Keep medicine names in English."),
        "🌻 Gujarati":   ("gu-IN", "Translate to very simple Gujarati. Short sentences. Keep medicine names in English."),
        "🌹 Bengali":    ("bn-IN", "Translate to very simple Bengali. Short sentences. Keep medicine names in English."),
        "🌾 Punjabi":    ("pa-IN", "Translate to very simple Punjabi (Gurmukhi). Short sentences. Keep medicine names in English."),
        "🏔️ Odia":       ("or-IN", "Translate to very simple Odia. Short sentences. Keep medicine names in English."),
        "🌴 Assamese":   ("as-IN", "Translate to very simple Assamese. Short sentences. Keep medicine names in English."),
    }

    with col_a:
        st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#ddeeff;margin-bottom:0.3rem">🌐 Choose Your Language / अपनी भाषा चुनें</p>', unsafe_allow_html=True)
        selected_lang = st.selectbox("Language", list(languages.keys()),
                                     key="elder_lang_select", label_visibility="collapsed")
    with col_b:
        st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#ddeeff;margin-bottom:0.3rem">&nbsp;</p>', unsafe_allow_html=True)
        if st.button("🌐  Translate Now", use_container_width=True, key="elder_translate_btn"):
            lang_tag, instruction = languages[selected_lang]
            st.session_state.voice_lang_tag = lang_tag
            st.session_state.voice_lang_label = selected_lang.split(" ", 1)[1]
            if instruction is None:
                st.session_state.voice_text = eng_full
                st.success("✅ English ready!")
            else:
                with st.spinner(f"Translating to {st.session_state.voice_lang_label}..."):
                    t_result, t_err = call_groq(
                        prompt=f"{instruction}\n\nText:\n{eng_full}", max_tokens=2048)
                if t_err:
                    st.error(f"Translation error: {t_err}")
                else:
                    st.session_state.voice_text = t_result
                    st.success(f"✅ Translated to {st.session_state.voice_lang_label}!")

    with col_c:
        st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#ddeeff;margin-bottom:0.3rem">&nbsp;</p>', unsafe_allow_html=True)
        if st.button("🧓  Simple Explanation", use_container_width=True, key="elder_explain_btn"):
            with st.spinner("Making it simple..."):
                simple_prompt = (
                    "Explain this prescription in very simple language for an elderly patient "
                    "who has no medical knowledge. Use short sentences, plain words, and bullet "
                    "points. Mention WHEN to take each medicine (morning/afternoon/night), WHAT "
                    "it is for, and any WARNING. Avoid all medical jargon.\n\n"
                    f"Prescription data:\n{json.dumps(r_v, indent=2)}"
                )
                simple_text, s_err = call_groq(prompt=simple_prompt, max_tokens=1500)
            if s_err:
                st.error(f"Error: {s_err}")
            else:
                st.session_state.elder_simple_text = simple_text

    # ── Simple Explanation Display ─────────────────────────────────────────
    if st.session_state.elder_simple_text:
        st.markdown(f"""
        <div style="background:#0d2a0d;border:2px solid #2a7a2a;border-radius:16px;
                    padding:1.4rem 1.8rem;margin:1rem 0;font-size:1.08rem;
                    color:#ccffcc;line-height:1.8;">
          <div style="font-size:1.25rem;font-weight:700;color:#66ff88;margin-bottom:0.7rem">
            🧓 Simple Guide for You
          </div>
          {st.session_state.elder_simple_text.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── BIG VOICE PLAYER (HTML component, full width) ───────────────────────
    voice_text_escaped = (
        st.session_state.voice_text
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", " ")
    )
    lang_tag  = st.session_state.voice_lang_tag
    lang_label = st.session_state.voice_lang_label

    # Build per-medicine JS data
    med_js_array = "["
    for i, m in enumerate(meds_v):
        name  = (m.get("name") or f"Medicine {i+1}").replace("`","").replace("$","")
        freq  = (m.get("frequency") or "as prescribed").replace("`","").replace("$","")
        dose  = (m.get("dosage") or "").replace("`","").replace("$","")
        ind   = (m.get("indication") or "").replace("`","").replace("$","")
        script = f"Medicine {i+1}: {name}. Dose: {dose}. Take: {freq}."
        if ind: script += f" This is for {ind}."
        script = script.replace("`","").replace("$","").replace("\\","")
        med_js_array += f'{{name:`{name}`,freq:`{freq}`,dose:`{dose}`,script:`{script}`}},'
    med_js_array += "]"

    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: transparent;
    color: #ffffff;
  }}

  /* ── Main player card ── */
  .player-card {{
    background: linear-gradient(145deg, #0b1a3a, #152850);
    border: 2.5px solid #2a5aa8;
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
  }}
  .player-title {{
    font-size: 1.6rem;
    font-weight: 800;
    color: #7ec8ff;
    margin-bottom: 0.3rem;
  }}
  .player-sub {{
    font-size: 1rem;
    color: #90a8c8;
    margin-bottom: 1.4rem;
  }}

  /* ── Voice selector row ── */
  .voice-selector-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.3rem;
    flex-wrap: wrap;
  }}
  .voice-selector-label {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #a0c4ff;
    white-space: nowrap;
  }}
  #voiceSelect {{
    background: #0e2a50;
    color: #ddeeff;
    border: 1.5px solid #2a5aa8;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    flex: 1;
    min-width: 220px;
    appearance: auto;
  }}
  #voiceSelect:focus {{ outline: 2px solid #3a9bd5; }}
  #voicePreviewBtn {{
    background: linear-gradient(135deg, #1a5fa8, #0e3d70);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s;
  }}
  #voicePreviewBtn:hover {{ background: linear-gradient(135deg, #2070c0, #1a5090); }}

  /* ── Big control buttons ── */
  .big-controls {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
  }}
  .big-btn {{
    border: none;
    border-radius: 16px;
    cursor: pointer;
    font-size: 1.35rem;
    font-weight: 800;
    padding: 1rem 2.2rem;
    min-width: 160px;
    transition: transform 0.12s, box-shadow 0.12s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
  }}
  .big-btn:active {{ transform: scale(0.95); }}
  #playBtn {{
    background: linear-gradient(135deg, #1a7a3a, #0f5a28);
    color: #ffffff;
    box-shadow: 0 4px 18px #1a7a3a60;
  }}
  #playBtn:hover {{ background: linear-gradient(135deg, #22a04a, #187030); }}
  #pauseBtn {{
    background: linear-gradient(135deg, #7a6a00, #554a00);
    color: #ffe066;
    box-shadow: 0 4px 18px #7a6a0040;
  }}
  #pauseBtn:hover {{ background: linear-gradient(135deg, #9a8800, #7a6a00); }}
  #stopBtn {{
    background: linear-gradient(135deg, #7a1a1a, #551010);
    color: #ffaaaa;
    box-shadow: 0 4px 18px #7a1a1a40;
  }}
  #stopBtn:hover {{ background: linear-gradient(135deg, #9a2020, #7a1a1a); }}

  /* ── Speed control ── */
  .speed-section {{
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
  }}
  .speed-label {{ font-size: 1.1rem; font-weight: 700; color: #a0c4ff; white-space: nowrap; }}
  input[type=range] {{
    accent-color: #3a9bd5;
    width: 220px;
    height: 8px;
    cursor: pointer;
  }}
  #speedVal {{
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
    min-width: 50px;
  }}

  /* ── Status bar ── */
  .status-bar {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c0d8ff;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    min-height: 3.2rem;
  }}
  .pulse-dot {{
    width: 14px; height: 14px; border-radius: 50%;
    background: #444; flex-shrink: 0;
  }}
  .pulse-dot.speaking {{ background: #44ff88; animation: blink 0.9s infinite; }}
  .pulse-dot.paused   {{ background: #ffcc00; }}
  @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}

  /* ── Per-medicine cards ── */
  .med-section-title {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #7ec8ff;
    margin: 0.5rem 0 1rem 0;
  }}
  .med-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
  }}
  .med-card {{
    background: linear-gradient(145deg, #0e2040, #172d55);
    border: 2px solid #2a4a88;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }}
  .med-name {{
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
  }}
  .med-detail {{
    font-size: 1rem;
    color: #90b8e8;
    line-height: 1.5;
  }}
  .med-play-btn {{
    margin-top: 0.6rem;
    background: linear-gradient(135deg, #1a5fa8, #0e3d70);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem;
    font-size: 1.1rem;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
    transition: background 0.15s;
  }}
  .med-play-btn:hover {{ background: linear-gradient(135deg, #2070c0, #1a5090); }}
  .med-play-btn:active {{ transform: scale(0.97); }}

  /* ── Alarm section ── */
  .alarm-card {{
    background: linear-gradient(145deg, #1a0d30, #2a1550);
    border: 2.5px solid #6633cc;
    border-radius: 22px;
    padding: 1.8rem 2rem;
    margin-top: 2rem;
  }}
  .alarm-title {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #cc88ff;
    margin-bottom: 0.4rem;
  }}
  .alarm-sub {{
    font-size: 0.95rem;
    color: #9966cc;
    margin-bottom: 1.2rem;
  }}
  .alarm-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.8rem;
    margin-bottom: 1rem;
  }}
  .alarm-row {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.7rem 1rem;
  }}
  .alarm-name {{ font-size: 1rem; font-weight: 700; color: #ddd; flex:1; }}
  input[type=time] {{
    background: #1a0a2a;
    color: #ffffff;
    border: 1.5px solid #5533aa;
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    cursor: pointer;
  }}
  #setAlarmsBtn {{
    background: linear-gradient(135deg, #6633cc, #4422aa);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2rem;
    font-size: 1.15rem;
    font-weight: 800;
    cursor: pointer;
    width: 100%;
    transition: background 0.15s;
    margin-top: 0.5rem;
  }}
  #setAlarmsBtn:hover {{ background: linear-gradient(135deg, #7744dd, #5533bb); }}
  #alarmStatus {{
    margin-top: 0.8rem;
    font-size: 1rem;
    color: #bb88ff;
    min-height: 1.4rem;
  }}
  .tip-box {{
    background: rgba(255,200,0,0.08);
    border: 1px solid #aa880030;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #ccaa44;
    font-size: 0.9rem;
    margin-top: 0.8rem;
  }}
</style>
</head>
<body>

<!-- ── BIG VOICE PLAYER ── -->
<div class="player-card">
  <div class="player-title">🔊 Medicine Voice Player</div>
  <div class="player-sub">Language: <strong style="color:#aaddff">{lang_label}</strong> &nbsp;·&nbsp; Choose a voice below, then press PLAY</div>

  <!-- ── Voice Dropdown ── -->
  <div class="voice-selector-row">
    <span class="voice-selector-label">🎙️ Choose Voice:</span>
    <select id="voiceSelect">
      <option value="">⏳ Loading voices...</option>
    </select>
    <button id="voicePreviewBtn" onclick="previewVoice()">▶ Preview Voice</button>
  </div>

  <div class="big-controls">
    <button class="big-btn" id="playBtn" onclick="startSpeech()">▶&nbsp; PLAY ALL</button>
    <button class="big-btn" id="pauseBtn" onclick="togglePause()">⏸&nbsp; PAUSE</button>
    <button class="big-btn" id="stopBtn" onclick="stopSpeech()">⏹&nbsp; STOP</button>
  </div>

  <div class="speed-section">
    <span class="speed-label">🐢 Reading Speed</span>
    <input type="range" id="speedSlider" min="0.4" max="1.2" step="0.1" value="0.7"
           oninput="speedVal.textContent=this.value+'×'">
    <span id="speedVal">0.7×</span>
    <span style="color:#6080a0;font-size:0.95rem">&nbsp;(Slow is better for elderly listeners)</span>
  </div>

  <div class="status-bar">
    <div class="pulse-dot" id="pulseDot"></div>
    <span id="statusText">Select a voice above, then press ▶ PLAY ALL to hear your medicines.</span>
  </div>
</div>

<!-- ── PER-MEDICINE CARDS ── -->
<div class="med-section-title">💊 Listen to Each Medicine One by One</div>
<div class="med-grid" id="medGrid"></div>

<!-- ── MEDICINE ALARM SETTER (at the bottom) ── -->
<div class="alarm-card">
  <div class="alarm-title">⏰ Set Medicine Reminder Alarms</div>
  <div class="alarm-sub">Set a time for each medicine. Your device will speak the medicine name when it is time to take it.</div>
  <div class="alarm-grid" id="alarmGrid"></div>
  <button id="setAlarmsBtn" onclick="setAlarms()">⏰ &nbsp;Set All Alarms Now</button>
  <div id="alarmStatus"></div>
  <div class="tip-box">💡 Keep this page open for alarms to work. Alarms will speak the medicine name in your chosen language using your selected voice.</div>
</div>

<script>
const FULL_TEXT = `{voice_text_escaped}`;
const LANG = "{lang_tag}";
const MEDS = {med_js_array};

let currentUtterance = null;
let alarmTimers = [];
let allVoices = [];
let selectedVoice = null;

// ── Voice loader & dropdown ──────────────────────────────────────────────
function populateVoiceDropdown() {{
  allVoices = window.speechSynthesis.getVoices();
  if (!allVoices.length) return;

  const sel = document.getElementById('voiceSelect');
  sel.innerHTML = '';

  // Preferred: voices matching the chosen language
  const langCode = LANG.split('-')[0];
  const matching = allVoices.filter(v => v.lang === LANG || v.lang.startsWith(langCode));
  const others   = allVoices.filter(v => v.lang !== LANG && !v.lang.startsWith(langCode));

  if (matching.length) {{
    const grp1 = document.createElement('optgroup');
    grp1.label = '✅ Best match for ' + LANG;
    matching.forEach((v, i) => {{
      const opt = document.createElement('option');
      opt.value = 'match_' + i;
      opt.textContent = v.name + ' (' + v.lang + ')' + (v.localService ? ' 📱' : ' ☁️');
      grp1.appendChild(opt);
    }});
    sel.appendChild(grp1);
  }}

  if (others.length) {{
    const grp2 = document.createElement('optgroup');
    grp2.label = '🌐 Other available voices';
    others.forEach((v, i) => {{
      const opt = document.createElement('option');
      opt.value = 'other_' + i;
      opt.textContent = v.name + ' (' + v.lang + ')' + (v.localService ? ' 📱' : ' ☁️');
      grp2.appendChild(opt);
    }});
    sel.appendChild(grp2);
  }}

  if (!allVoices.length) {{
    sel.innerHTML = '<option value="">⚠️ No voices found on this device</option>';
  }}

  // Default select first matching voice
  sel.selectedIndex = 0;
  updateSelectedVoice();
  setStatus('', 'Voice ready. Press ▶ PLAY ALL to hear your medicines.');
}}

function updateSelectedVoice() {{
  const sel = document.getElementById('voiceSelect');
  const val = sel.value;
  if (!val) {{ selectedVoice = null; return; }}
  if (val.startsWith('match_')) {{
    const idx = parseInt(val.split('_')[1]);
    const langCode = LANG.split('-')[0];
    const matching = allVoices.filter(v => v.lang === LANG || v.lang.startsWith(langCode));
    selectedVoice = matching[idx] || null;
  }} else {{
    const idx = parseInt(val.split('_')[1]);
    const langCode = LANG.split('-')[0];
    const others = allVoices.filter(v => v.lang !== LANG && !v.lang.startsWith(langCode));
    selectedVoice = others[idx] || null;
  }}
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('voiceSelect').addEventListener('change', updateSelectedVoice);
}});

function previewVoice() {{
  updateSelectedVoice();
  const rate = parseFloat(document.getElementById('speedSlider').value);
  speakWith('Hello. This is a voice preview. I will read your medicines in this voice.', rate, null);
}}

// ── Voice player ──────────────────────────────────────────────────────────
function setStatus(dotClass, msg) {{
  document.getElementById('pulseDot').className = 'pulse-dot ' + dotClass;
  document.getElementById('statusText').textContent = msg;
}}

function speakWith(text, rate, onEnd) {{
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = LANG;
  u.rate = rate || 0.7;
  if (selectedVoice) u.voice = selectedVoice;
  if (onEnd) u.onend = onEnd;
  u.onerror = e => setStatus('', '⚠️ Voice error: ' + e.error);
  currentUtterance = u;
  window.speechSynthesis.speak(u);
}}

function startSpeech() {{
  updateSelectedVoice();
  const rate = parseFloat(document.getElementById('speedSlider').value);
  setStatus('speaking', 'Speaking all medicines... You can press PAUSE anytime.');
  speakWith(FULL_TEXT, rate, () => setStatus('', '✅ Finished! Press ▶ PLAY ALL again to replay.'));
}}

function togglePause() {{
  if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {{
    window.speechSynthesis.pause();
    setStatus('paused', 'Paused. Press PAUSE button again to continue.');
    document.getElementById('pauseBtn').innerHTML = '▶&nbsp; RESUME';
  }} else if (window.speechSynthesis.paused) {{
    window.speechSynthesis.resume();
    setStatus('speaking', 'Resumed...');
    document.getElementById('pauseBtn').innerHTML = '⏸&nbsp; PAUSE';
  }}
}}

function stopSpeech() {{
  window.speechSynthesis.cancel();
  document.getElementById('pauseBtn').innerHTML = '⏸&nbsp; PAUSE';
  setStatus('', 'Stopped. Press ▶ PLAY ALL to start again from the beginning.');
}}

// ── Per-medicine cards ────────────────────────────────────────────────────
function buildMedCards() {{
  const grid = document.getElementById('medGrid');
  if (!MEDS.length) {{
    grid.innerHTML = '<p style="color:#6080a0">No medicines found in analysis.</p>';
    return;
  }}
  MEDS.forEach((m, i) => {{
    const card = document.createElement('div');
    card.className = 'med-card';
    card.innerHTML = `
      <div class="med-name">💊 ${{m.name}}</div>
      <div class="med-detail">📏 Dose: <strong>${{m.dose || '—'}}</strong></div>
      <div class="med-detail">🕐 When: <strong>${{m.freq || '—'}}</strong></div>
      <button class="med-play-btn" onclick="playMed(${{i}})">
        🔊 &nbsp;Listen to This Medicine
      </button>
    `;
    grid.appendChild(card);
  }});
}}

function playMed(i) {{
  updateSelectedVoice();
  const rate = parseFloat(document.getElementById('speedSlider').value);
  const m = MEDS[i];
  setStatus('speaking', 'Playing: ' + m.name);
  speakWith(m.script, rate, () => setStatus('', '✅ Done. Choose another medicine or press ▶ PLAY ALL for all.'));
}}

// ── Alarm setter ──────────────────────────────────────────────────────────
function buildAlarmGrid() {{
  const grid = document.getElementById('alarmGrid');
  if (!MEDS.length) {{
    grid.innerHTML = '<p style="color:#9966cc">No medicines to set alarms for.</p>';
    return;
  }}
  MEDS.forEach((m, i) => {{
    const row = document.createElement('div');
    row.className = 'alarm-row';
    row.innerHTML = `
      <div class="alarm-name">💊 ${{m.name}}</div>
      <input type="time" id="alarm_${{i}}" value="">
    `;
    grid.appendChild(row);
  }});
}}

function setAlarms() {{
  alarmTimers.forEach(t => clearTimeout(t));
  alarmTimers = [];
  let count = 0;

  MEDS.forEach((m, i) => {{
    const timeEl = document.getElementById('alarm_' + i);
    if (!timeEl || !timeEl.value) return;
    const [hh, mm] = timeEl.value.split(':').map(Number);
    const now = new Date();
    const alarm = new Date();
    alarm.setHours(hh, mm, 0, 0);
    if (alarm <= now) alarm.setDate(alarm.getDate() + 1);
    const delay = alarm - now;
    const timer = setTimeout(() => {{
      const msg = `Time to take your medicine: ${{m.name}}. Dose: ${{m.dose}}. ${{m.script}}`;
      speakWith(msg, 0.75, null);
      document.getElementById('alarmStatus').textContent = '🔔 ALARM: Time for ' + m.name + '!';
      if (Notification && Notification.permission === 'granted') {{
        new Notification('💊 Medicine Time!', {{
          body: 'Time to take: ' + m.name + ' — ' + m.dose,
          icon: ''
        }});
      }}
    }}, delay);
    alarmTimers.push(timer);
    count++;
  }});

  if (Notification && Notification.permission === 'default') {{
    Notification.requestPermission();
  }}

  if (count > 0) {{
    document.getElementById('alarmStatus').textContent =
      '✅ ' + count + ' alarm(s) set! This page will speak your medicine name at the right time.';
  }} else {{
    document.getElementById('alarmStatus').textContent =
      '⚠️ Please pick a time for at least one medicine above.';
  }}
}}

// ── Init ──────────────────────────────────────────────────────────────────
window.speechSynthesis.onvoiceschanged = populateVoiceDropdown;
// Also try immediately in case voices are already loaded
if (window.speechSynthesis.getVoices().length > 0) populateVoiceDropdown();

buildMedCards();
buildAlarmGrid();
</script>
</body>
</html>
""", height=1080, scrolling=True)


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
