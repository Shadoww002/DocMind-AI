"""
app.py — Streamlit frontend
Changes from previous version:
  1. Entity chip rendering: reads scored entity lists from extractor
     and shows a confidence badge on each chip
  2. Export button: st.download_button packages full session as JSON
"""
import json
import time
import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1500px; }

    .kpi-box { background:#111726; border:1px solid #1f293d; padding:20px; border-radius:12px; }
    .kpi-val { font-size:1.6rem; font-weight:700; color:#38bdf8; margin-bottom:4px; }
    .kpi-lbl { font-size:0.8rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em; }

    .short-summary-box { background:#111827; border-left:4px solid #38bdf8; padding:22px; border-radius:8px; margin-bottom:24px; font-size:1.05rem; color:#f3f4f6; line-height:1.7; }
    .long-summary-box  { background:#0b0f19; padding:26px; border-radius:8px; border:1px solid #1f293d; color:#d1d5db; line-height:1.8; font-size:1rem; }

    .domain-card { background:#121824; border-left:4px solid #38bdf8; padding:18px; border-radius:0 12px 12px 0; margin-bottom:18px; border-top:1px solid #1f293d; border-right:1px solid #1f293d; border-bottom:1px solid #1f293d; }
    .domain-card.medical-alert  { border-left-color:#ef4444; background:#1a1013; }
    .domain-card.legal-warning  { border-left-color:#f59e0b; background:#1a1410; }
    .domain-card.resume-success { border-left-color:#10b981; background:#0f1815; }
    .card-title { font-weight:700; font-size:0.85rem; color:#f3f4f6; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em; }

    /* ── CHANGE 1: entity chip styles ── */
    .chips { display:flex; flex-wrap:wrap; gap:6px; }
    .chip {
        display:inline-flex; align-items:center; gap:5px;
        padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500;
        border:0.5px solid rgba(255,255,255,0.12);
        background:rgba(255,255,255,0.05); color:#e2e8f0;
    }
    .chip-score {
        font-size:10px; padding:1px 5px; border-radius:4px;
        font-weight:600; letter-spacing:0.03em;
    }
    .score-high   { background:rgba(16,185,129,0.2); color:#34d399; }
    .score-medium { background:rgba(245,158,11,0.2); color:#fbbf24; }
    .score-low    { background:rgba(239,68,68,0.15);  color:#f87171; }

    .meta-bar { background:#0d1220; border:1px solid #1f293d; border-radius:8px; padding:10px 16px; font-size:0.8rem; color:#6b7280; margin-bottom:1.5rem; display:flex; gap:24px; flex-wrap:wrap; }
    .meta-item strong { color:#9ca3af; }

    .stTabs [data-baseweb="tab-list"] { gap:1rem; background-color:transparent; padding:0; border-bottom:1px solid #1f293d; }
    .stTabs [data-baseweb="tab"] { height:46px; background-color:transparent; padding:0 22px; font-size:1rem; font-weight:500; color:#6b7280; }
    .stTabs [aria-selected="true"] { color:#38bdf8 !important; font-weight:600; border-bottom:2px solid #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "app_state": "upload", "doc_id": None, "active_domain": None,
    "processed_payload": None, "chat_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_app():
    # FIX 6: delete document vectors from ChromaDB before clearing session
    # Without this, every "New Document" session leaves orphaned chunks in the DB.
    # On a local machine with many test runs this grows unbounded.
    doc_id = st.session_state.get("doc_id")
    domain = st.session_state.get("active_domain")
    if doc_id and domain:
        try:
            requests.delete(
                f"{BACKEND_URL}/document",
                json={"document_id": doc_id, "domain": domain},
                timeout=10,
            )
        except Exception:
            pass  # non-critical — vectors will be overwritten on next upsert

    for k in ["doc_id", "active_domain", "processed_payload"]:
        st.session_state[k] = None
    st.session_state.app_state = "upload"
    st.session_state.chat_log = []


# ── CHANGE 1: chip renderer ───────────────────────────────────────────────────

def _score_class(score: float) -> str:
    if score >= 0.90: return "score-high"
    if score >= 0.80: return "score-medium"
    return "score-low"

def _score_label(score: float) -> str:
    # FIX 7: all branches were identical — simplified to one line
    return f"{int(score * 100)}%"

def render_field(field_value) -> str:
    """
    Renders a field value as HTML.
    - If it's a list of {"value":..., "score":...} dicts → chip row with badges
    - If it's a plain string → plain text
    """
    if isinstance(field_value, list) and field_value:
        if isinstance(field_value[0], dict) and "value" in field_value[0]:
            chips_html = ""
            for entity in field_value:
                val   = entity.get("value", "")
                score = entity.get("score", 0.75)
                sc    = _score_class(score)
                lbl   = _score_label(score)
                chips_html += (
                    f'<span class="chip">'
                    f'{val}'
                    f'<span class="chip-score {sc}">{lbl}</span>'
                    f'</span>'
                )
            return f'<div class="chips">{chips_html}</div>'
    # plain string fallback
    val = str(field_value) if field_value else "Not found."
    if "HIGH RISK" in val:
        val = val.replace("HIGH RISK", "<strong style='color:#f59e0b'>HIGH RISK</strong>")
    elif "MODERATE RISK" in val:
        val = val.replace("MODERATE RISK", "<strong style='color:#fbbf24'>MODERATE RISK</strong>")
    return f'<span style="color:#d1d5db;font-size:1rem;line-height:1.6;">{val}</span>'


# ── CHANGE 2: export builder ──────────────────────────────────────────────────

def build_export(payload: dict, domain: str, chat_log: list) -> str:
    """
    Packages the full session into a clean JSON export.
    Scored entity lists are flattened to "value (score%)" strings for readability.
    """
    def flatten_field(v):
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [f"{e['value']} ({int(e['score']*100)}%)" for e in v]
        return v

    extracted = {
        k: flatten_field(v)
        for k, v in payload.get("extracted_data", {}).items()
    }

    export = {
        "docmind_ai_export": True,
        "document": {
            "filename":    payload.get("filename"),
            "document_id": payload.get("document_id"),
            "domain":      domain,
            "page_count":  payload.get("page_count"),
        },
        "summary": {
            "short":    payload.get("short_summary"),
            "detailed": payload.get("detailed_summary"),
        },
        "extracted_data": extracted,
        "chat_history": [
            {"role": e["role"], "text": e["text"]}
            for e in chat_log
        ],
        "processing_meta": payload.get("processing_meta", {}),
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_state == "upload":
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown("<h1 style='text-align:center;color:#38bdf8;font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;margin-bottom:4px;'>🧠 DocMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6b7280;font-size:1.05rem;margin-bottom:2.5rem;'>Multi-Domain Document Intelligence</p>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("**1. Choose Domain**")
            selected_domain = st.selectbox(
                "Domain",
                options=["medical", "legal", "resume"],
                format_func=lambda x: {
                    "medical": "🏥  Medical Records & Clinical Reports",
                    "legal":   "⚖️  Indian Legal Contracts & Deeds",
                    "resume":  "📄  Resumes & Professional CVs",
                }[x],
                label_visibility="collapsed",
            )
            st.divider()
            st.markdown("**2. Extraction Parameters**")
            params_dict = {}
            detail_map = {"Concise": 1, "Standard": 1.5, "Comprehensive": 2}

            if selected_domain == "medical":
                c1, c2 = st.columns(2)
                params_dict["extract_dosages"] = c1.checkbox("Extract Dosages & Metrics", value=True)
                params_dict["map_pathology"]   = c1.checkbox("Map Pathological History", value=True)
                params_dict["anonymize_pii"]   = c2.checkbox("HIPAA Masking", value=False)
                params_dict["ner_threshold"]   = c2.slider("NER Confidence Threshold", 50, 99, 80)
                params_dict["summary_detail"]  = detail_map[st.select_slider("Summary Detail", ["Concise","Standard","Comprehensive"], "Standard")]
            elif selected_domain == "legal":
                c1, c2 = st.columns(2)
                params_dict["isolate_signees"]  = c1.checkbox("Isolate Signees", value=True)
                params_dict["scan_liabilities"] = c1.checkbox("Scan Liabilities", value=True)
                params_dict["anonymize_pii"]    = c2.checkbox("Redact PII", value=False)
                params_dict["risk_sensitivity"] = c2.slider("Risk Sensitivity", 1, 10, 7)
                params_dict["summary_detail"]   = detail_map[st.select_slider("Summary Detail", ["Concise","Standard","Comprehensive"], "Standard")]
            elif selected_domain == "resume":
                c1, c2 = st.columns(2)
                params_dict["strict_skills"]  = c1.checkbox("Strict Skill Matching", value=True)
                params_dict["parse_academic"] = c1.checkbox("Academic Timeline", value=True)
                params_dict["anonymize_pii"]  = c2.checkbox("Blind Screening", value=False)
                params_dict["target_role"]    = c2.text_input("Target Role", placeholder="e.g. ML Engineer")
                params_dict["summary_detail"] = detail_map[st.select_slider("Summary Detail", ["Concise","Standard","Comprehensive"], "Standard")]

            st.divider()
            st.markdown("**3. Upload PDF**")
            uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
            if uploaded_file and uploaded_file.size > 10 * 1024 * 1024:
                st.warning("⚠️ Large file (>10 MB). May be slow on CPU.")

            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
            if st.button("🚀 Analyse Document", use_container_width=True, type="primary"):
                if not uploaded_file:
                    st.error("Please upload a PDF first.")
                else:
                    with st.status("Processing…", expanded=True) as status:
                        st.write("Parsing PDF and building vector index…")
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data  = {"domain": selected_domain, "params": json.dumps(params_dict)}
                        try:
                            resp = requests.post(f"{BACKEND_URL}/process", files=files, data=data, timeout=300)
                            if resp.status_code == 200:
                                st.session_state.processed_payload = resp.json()
                                st.session_state.doc_id            = st.session_state.processed_payload["document_id"]
                                st.session_state.active_domain     = selected_domain
                                st.session_state.app_state         = "dashboard"
                                status.update(label="Done!", state="complete", expanded=False)
                                time.sleep(0.2)
                                st.rerun()
                            else:
                                status.update(label="Failed", state="error", expanded=True)
                                try:    detail = resp.json().get("detail", resp.text)
                                except: detail = resp.text
                                st.error(f"Backend error ({resp.status_code}): {detail}")
                        except requests.exceptions.Timeout:
                            status.update(label="Timeout", state="error", expanded=True)
                            st.error("Request timed out. Try Concise mode or a shorter document.")
                        except Exception as e:
                            status.update(label="Connection error", state="error", expanded=True)
                            st.error(f"Could not reach backend: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.app_state == "dashboard":
    payload          = st.session_state.processed_payload
    domain           = st.session_state.active_domain
    extracted_fields = payload.get("extracted_data", {})
    meta             = payload.get("processing_meta", {})

    with st.sidebar:
        st.markdown("### 📁 Session")
        st.caption("File")
        st.code(payload.get("filename", "—"), language="text")
        st.caption("Domain")
        st.info(domain.upper())
        st.caption(f"Pages: **{payload.get('page_count','—')}** · Chunks: **{payload.get('chunk_count','—')}**")

        st.divider()

        # ── CHANGE 2: Export button ───────────────────────────────────────────
        export_json = build_export(payload, domain, st.session_state.chat_log)
        filename    = f"docmind_{domain}_{st.session_state.doc_id}.json"
        st.download_button(
            label="⬇ Export session as JSON",
            data=export_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
        )
        # ─────────────────────────────────────────────────────────────────────

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        if st.button("↩ New Document", use_container_width=True):
            reset_app()
            st.rerun()

    st.markdown(f"<h1 style='letter-spacing:-0.03em;font-weight:800;margin-bottom:4px;'>{domain.capitalize()} Document Analysis</h1>", unsafe_allow_html=True)

    if meta:
        elapsed = meta.get("elapsed_seconds", "—")
        device  = meta.get("device", "—")
        model   = meta.get("model", "—").split("/")[-1]
        chunks  = meta.get("chunks_processed", "—")
        st.markdown(
            f"<div class='meta-bar'>"
            f"<span><strong>Model:</strong> {model}</span>"
            f"<span><strong>Device:</strong> {device}</span>"
            f"<span><strong>Chunks:</strong> {chunks}</span>"
            f"<span><strong>Time:</strong> {elapsed}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-box"><div class="kpi-val">{domain.upper()}</div><div class="kpi-lbl">Domain</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-box"><div class="kpi-val">{payload.get("page_count","—")}</div><div class="kpi-lbl">Pages</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-box"><div class="kpi-val">{len(extracted_fields)}</div><div class="kpi-lbl">Fields</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-box"><div class="kpi-val">{st.session_state.doc_id[-6:].upper()}</div><div class="kpi-lbl">Vector ID</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    tab_summary, tab_analytics, tab_rag = st.tabs(["📋 Summary", "📊 Extracted Fields", "💬 Chat"])

    with tab_summary:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Brief Insight")
        st.markdown(f"<div class='short-summary-box'>{payload.get('short_summary','—')}</div>", unsafe_allow_html=True)
        st.markdown("##### Detailed Breakdown")
        detailed = payload.get("detailed_summary", "")
        formatted = detailed.replace("• ", "<li>").replace("\n\n", "</li>")
        if "<li>" in formatted:
            formatted = f"<ul style='padding-left:1.2rem;line-height:1.9'>{formatted}</li></ul>"
        st.markdown(f"<div class='long-summary-box'>{formatted}</div>", unsafe_allow_html=True)

    with tab_analytics:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # ── CHANGE 2: export in analytics tab too ────────────────────────────
        col_export, _ = st.columns([1, 3])
        with col_export:
            st.download_button(
                label="⬇ Export JSON",
                data=export_json,
                file_name=filename,
                mime="application/json",
            )
        # ─────────────────────────────────────────────────────────────────────

        if not extracted_fields:
            st.info("No fields extracted.")
        else:
            col_l, col_r = st.columns(2, gap="large")
            for idx, (field, content) in enumerate(extracted_fields.items()):
                target_col = col_l if idx % 2 == 0 else col_r
                with target_col:
                    style = "domain-card"
                    if domain == "medical" and field in ("Diagnoses & Conditions", "Medications & Dosages", "Lab Values & Vitals"):
                        style += " medical-alert"
                    elif domain == "legal" and field in ("Risks & Red Flags", "Financial Liabilities", "Monetary Amounts"):
                        style += " legal-warning"
                    elif domain == "resume" and field in ("Target Role Alignment", "Skills Matrix"):
                        style += " resume-success"

                    # ── CHANGE 1: use render_field for chip rendering ─────────
                    body_html = render_field(content)
                    # ─────────────────────────────────────────────────────────

                    st.markdown(
                        f"<div class='{style}'>"
                        f"<div class='card-title'>{field}</div>"
                        f"{body_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    with tab_rag:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.caption("Questions answered strictly from the document's vector index.")
        chat_frame = st.container(height=460, border=True)
        with chat_frame:
            if not st.session_state.chat_log:
                st.markdown("<p style='text-align:center;color:#4b5563;margin-top:140px;'>Vector index ready — ask below.</p>", unsafe_allow_html=True)
            else:
                for entry in st.session_state.chat_log:
                    with st.chat_message(entry["role"]):
                        st.markdown(entry["text"])
                        for cite in entry.get("citations", []):
                            st.markdown(f"**Source:** `Page {cite['page']}`")
                            if cite.get("text_snippet"):
                                st.caption(f"_{cite['text_snippet']}_")

        query = st.chat_input("Ask a question about this document…")
        if query:
            st.session_state.chat_log.append({"role": "user", "text": query, "citations": []})
            with chat_frame:
                with st.chat_message("user"):
                    st.markdown(query)
                with st.chat_message("assistant"):
                    with st.spinner("Searching vector index…"):
                        try:
                            resp = requests.post(
                                f"{BACKEND_URL}/query",
                                json={"document_id": st.session_state.doc_id, "domain": domain, "question": query},
                                timeout=120,
                            )
                            if resp.status_code == 200:
                                ans         = resp.json()
                                answer_text = ans.get("answer", "No answer returned.")
                                citations   = ans.get("citations", [])
                                confidence  = ans.get("confidence", 0.0)
                                st.markdown(answer_text)
                                if confidence:
                                    st.caption(f"Confidence: {int(confidence * 100)}%")
                                for cite in citations:
                                    st.markdown(f"**Source:** `Page {cite['page']}`")
                                    if cite.get("text_snippet"):
                                        st.caption(f"_{cite['text_snippet']}_")
                                st.session_state.chat_log.append({
                                    "role": "assistant", "text": answer_text,
                                    "citations": citations,
                                })
                            else:
                                try:    detail = resp.json().get("detail", resp.text)
                                except: detail = resp.text
                                st.error(f"Query failed ({resp.status_code}): {detail}")
                        except requests.exceptions.Timeout:
                            st.error("Timed out. Try a shorter question.")
                        except Exception as e:
                            st.error(f"Connection error: {e}")