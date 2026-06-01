import streamlit as st
import requests
import time
import json

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

# ── Page config ───────────────────────────────────────────────────────────────
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
    .card-title { font-weight:700; font-size:0.85rem; color:#f3f4f6; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em; }
    .card-body  { color:#d1d5db; font-size:1rem; line-height:1.6; }

    /* IMPROVEMENT: meta bar for processing info */
    .meta-bar { background:#0d1220; border:1px solid #1f293d; border-radius:8px; padding:10px 16px; font-size:0.8rem; color:#6b7280; margin-bottom:1.5rem; display:flex; gap:24px; }
    .meta-item strong { color:#9ca3af; }

    .stTabs [data-baseweb="tab-list"] { gap:1rem; background-color:transparent; padding:0; border-bottom:1px solid #1f293d; }
    .stTabs [data-baseweb="tab"] { height:46px; background-color:transparent; padding:0 22px; font-size:1rem; font-weight:500; color:#6b7280; }
    .stTabs [aria-selected="true"] { color:#38bdf8 !important; font-weight:600; border-bottom:2px solid #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "app_state": "upload",
    "doc_id": None,
    "active_domain": None,
    "processed_payload": None,
    "chat_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_app():
    for key in ["doc_id", "active_domain", "processed_payload"]:
        st.session_state[key] = None
    st.session_state.app_state = "upload"
    st.session_state.chat_log = []


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_state == "upload":
    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        st.markdown("<h1 style='text-align:center;color:#38bdf8;font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;margin-bottom:4px;'>🧠 DocMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6b7280;font-size:1.05rem;margin-bottom:2.5rem;'>Multi-Domain Document Intelligence</p>", unsafe_allow_html=True)

        with st.container(border=True):

            # ── Domain selector ──────────────────────────────────────────────
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

            # ── Parameters ───────────────────────────────────────────────────
            st.markdown("**2. Extraction Parameters**")
            params_dict = {}
            detail_map = {"Concise": 1, "Standard": 1.5, "Comprehensive": 2}

            if selected_domain == "medical":
                c1, c2 = st.columns(2)
                params_dict["extract_dosages"]  = c1.checkbox("Extract Dosages & Metrics", value=True)
                params_dict["map_pathology"]     = c1.checkbox("Map Pathological History", value=True)
                params_dict["anonymize_pii"]     = c2.checkbox("HIPAA Masking (Anonymize PII)", value=False)
                params_dict["ner_threshold"]     = c2.slider("NER Confidence Threshold", 50, 99, 80)
                choice = st.select_slider("Summary Detail", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"]    = detail_map[choice]

            elif selected_domain == "legal":
                c1, c2 = st.columns(2)
                params_dict["isolate_signees"]   = c1.checkbox("Isolate Signees & Parties", value=True)
                params_dict["scan_liabilities"]  = c1.checkbox("Scan Financial Liabilities", value=True)
                params_dict["anonymize_pii"]     = c2.checkbox("Redact Corporate PII", value=False)
                params_dict["risk_sensitivity"]  = c2.slider("Risk Sensitivity", 1, 10, 7)
                choice = st.select_slider("Summary Detail", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"]    = detail_map[choice]

            elif selected_domain == "resume":
                c1, c2 = st.columns(2)
                params_dict["strict_skills"]     = c1.checkbox("Strict Skill Matching", value=True)
                params_dict["parse_academic"]    = c1.checkbox("Academic Timeline Parse", value=True)
                params_dict["anonymize_pii"]     = c2.checkbox("Blind Screening Mode", value=False)
                params_dict["target_role"]       = c2.text_input("Target Role (optional)", placeholder="e.g. ML Engineer")
                choice = st.select_slider("Summary Detail", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"]    = detail_map[choice]

            st.divider()

            # ── File upload ──────────────────────────────────────────────────
            st.markdown("**3. Upload PDF**")
            uploaded_file = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")

            # IMPROVEMENT: warn if file looks very large (>10 MB may be slow on CPU)
            if uploaded_file and uploaded_file.size > 10 * 1024 * 1024:
                st.warning("⚠️ Large file detected (>10 MB). Processing may take several minutes on CPU.")

            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

            if st.button("🚀 Analyse Document", use_container_width=True, type="primary"):
                if not uploaded_file:
                    st.error("Please upload a PDF file first.")
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
                                # IMPROVEMENT: show the actual backend detail message
                                try:
                                    detail = resp.json().get("detail", resp.text)
                                except Exception:
                                    detail = resp.text
                                st.error(f"Backend error ({resp.status_code}): {detail}")
                        except requests.exceptions.Timeout:
                            status.update(label="Timeout", state="error", expanded=True)
                            st.error("Request timed out (5 min). Try a shorter document or Concise summary mode.")
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

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Session")
        st.caption("File")
        st.code(payload.get("filename", "—"), language="text")
        st.caption("Domain")
        st.info(domain.upper())

        # IMPROVEMENT: show page/chunk counts in sidebar
        page_count  = payload.get("page_count", "—")
        chunk_count = payload.get("chunk_count", "—")
        st.caption(f"Pages parsed: **{page_count}** · Chunks: **{chunk_count}**")

        st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
        if st.button("↩ New Document", use_container_width=True):
            reset_app()
            st.rerun()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"<h1 style='letter-spacing:-0.03em;font-weight:800;margin-bottom:4px;'>{domain.capitalize()} Document Analysis</h1>", unsafe_allow_html=True)

    # IMPROVEMENT: processing meta bar (model, device, time)
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

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-box"><div class="kpi-val">{domain.upper()}</div><div class="kpi-lbl">Domain</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-box"><div class="kpi-val">{payload.get("page_count","—")}</div><div class="kpi-lbl">Pages</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-box"><div class="kpi-val">{len(extracted_fields)}</div><div class="kpi-lbl">Fields Extracted</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-box"><div class="kpi-val">{st.session_state.doc_id[-6:].upper()}</div><div class="kpi-lbl">Vector ID</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_summary, tab_analytics, tab_rag = st.tabs(["📋 Summary", "📊 Extracted Fields", "💬 Chat"])

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Brief Insight")
        short = payload.get("short_summary") or "No summary generated."
        st.markdown(f"<div class='short-summary-box'>{short}</div>", unsafe_allow_html=True)

        st.markdown("##### Detailed Breakdown")
        detailed = payload.get("detailed_summary") or "No detailed breakdown available."
        # IMPROVEMENT: render bullet points as proper HTML list instead of raw •
        formatted = detailed.replace("• ", "<li>").replace("\n\n", "</li>")
        if "<li>" in formatted:
            formatted = f"<ul style='padding-left:1.2rem;line-height:1.9'>{formatted}</li></ul>"
        st.markdown(f"<div class='long-summary-box'>{formatted}</div>", unsafe_allow_html=True)

    # ── Tab 2: Extracted fields ───────────────────────────────────────────────
    with tab_analytics:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        if not extracted_fields:
            st.info("No fields were extracted from this document.")
        else:
            # IMPROVEMENT: show a quick copy-all button
            with st.expander("📋 Export all extracted fields as JSON"):
                st.json(extracted_fields)

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

                    # IMPROVEMENT: highlight risk keywords in legal cards
                    display_content = content
                    if domain == "legal" and field == "Risks & Red Flags":
                        for kw in ("HIGH RISK", "MODERATE RISK"):
                            if kw in content:
                                display_content = content.replace(kw, f"<strong style='color:#f59e0b'>{kw}</strong>")

                    st.markdown(
                        f"<div class='{style}'>"
                        f"<div class='card-title'>{field}</div>"
                        f"<div class='card-body'>{display_content}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Tab 3: RAG chat ───────────────────────────────────────────────────────
    with tab_rag:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.caption("Questions are answered strictly from the document's vector index.")

        chat_frame = st.container(height=460, border=True)

        with chat_frame:
            if not st.session_state.chat_log:
                st.markdown(
                    "<p style='text-align:center;color:#4b5563;margin-top:140px;'>"
                    "Vector index ready — ask a question below.</p>",
                    unsafe_allow_html=True,
                )
            else:
                for entry in st.session_state.chat_log:
                    with st.chat_message(entry["role"]):
                        st.markdown(entry["text"])
                        for cite in entry.get("citations", []):
                            st.markdown(f"**Source:** `Page {cite['page']}`")
                            st.caption(f"_{cite.get('text_snippet', '')}_")

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
                                json={
                                    "document_id": st.session_state.doc_id,
                                    "domain":       domain,
                                    "question":     query,
                                },
                                timeout=120,
                            )
                            if resp.status_code == 200:
                                ans = resp.json()
                                answer_text = ans.get("answer", "No answer returned.")
                                citations   = ans.get("citations", [])
                                st.markdown(answer_text)
                                for cite in citations:
                                    st.markdown(f"**Source:** `Page {cite['page']}`")
                                    st.caption(f"_{cite.get('text_snippet','')}_")
                                st.session_state.chat_log.append({
                                    "role": "assistant",
                                    "text": answer_text,
                                    "citations": citations,
                                })
                            else:
                                # IMPROVEMENT: surface actual error from backend
                                try:
                                    detail = resp.json().get("detail", resp.text)
                                except Exception:
                                    detail = resp.text
                                st.error(f"Query failed ({resp.status_code}): {detail}")
                        except requests.exceptions.Timeout:
                            st.error("Query timed out. Try a shorter or more specific question.")
                        except Exception as e:
                            st.error(f"Connection error: {e}")