import streamlit as st
import requests
import time
import json

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

# ==========================================
# PAGE CONFIG & ULTRA-PREMIUM CLEAN CSS
# ==========================================
st.set_page_config(
    page_title="DocMind AI Enterprise", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Premium spacious design language injection
st.markdown("""
<style>
    /* Deep space background with improved tracking */
    .stApp { background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    
    /* Spacious container pad */
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1500px; }
    
    /* Clean, uncluttered custom layout blocks */
    .kpi-box {
        background: #111726;
        border: 1px solid #1f293d;
        padding: 24px;
        border-radius: 12px;
        text-align: left;
    }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }
    .kpi-lbl { font-size: 0.85rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Summary Vertical Stacking */
    .short-summary-box { background: #111827; border-left: 4px solid #38bdf8; padding: 24px; border-radius: 8px; margin-bottom: 30px; font-size: 1.15rem; color: #f3f4f6; line-height: 1.6;}
    .long-summary-box { background: #0b0f19; padding: 30px; border-radius: 8px; border: 1px solid #1f293d; color: #d1d5db; line-height: 1.8; font-size: 1.05rem;}

    /* Domain Adaptive Cards */
    .domain-card {
        background: #121824;
        border-left: 4px solid #38bdf8;
        padding: 20px;
        border-radius: 0px 12px 12px 0px;
        margin-bottom: 20px;
        border-top: 1px solid #1f293d;
        border-right: 1px solid #1f293d;
        border-bottom: 1px solid #1f293d;
    }
    .domain-card.medical-alert { border-left-color: #ef4444; background: #1a1013; }
    .domain-card.legal-warning { border-left-color: #f59e0b; background: #1a1410; }
    .domain-card.resume-success { border-left-color: #10b981; background: #0f1815; }
    
    .card-title { font-weight: 700; font-size: 1.05rem; color: #f3f4f6; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 0.05em;}
    .card-body { color: #d1d5db; font-size: 1.05rem; line-height: 1.6; }

    /* Modern clean tab system */
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; background-color: transparent; padding: 0; border-bottom: 1px solid #1f293d; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; background-color: transparent; padding: 0px 24px;
        font-size: 1.05rem; font-weight: 500; color: #6b7280;
    }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; font-weight: 600; border-bottom: 2px solid #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT
# ==========================================
if "app_state" not in st.session_state:
    st.session_state.app_state = "upload"
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "active_domain" not in st.session_state:
    st.session_state.active_domain = None
if "processed_payload" not in st.session_state:
    st.session_state.processed_payload = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

def reset_app():
    st.session_state.app_state = "upload"
    st.session_state.doc_id = None
    st.session_state.active_domain = None
    st.session_state.processed_payload = None
    st.session_state.chat_log = []

# ==========================================
# SCREEN 1: ROOMY UPLOAD CENTER & PARAMS
# ==========================================
if st.session_state.app_state == "upload":
    _, col_center, _ = st.columns([1, 1.8, 1])
    
    with col_center:
        st.markdown("<h1 style='text-align: center; color: #38bdf8; font-size: 2.8rem; font-weight: 800; letter-spacing:-0.03em; margin-bottom: 6px;'>🧠 DocMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280; font-size: 1.15rem; margin-bottom: 3rem;'>Enterprise Multi-Domain Document Intelligence</p>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p style='font-weight:600; font-size:1.1rem; margin-bottom:8px;'>1. Target Intelligence Model</p>", unsafe_allow_html=True)
            selected_domain = st.selectbox(
                "Select Intelligence Domain Profile",
                options=["medical", "legal", "resume"],
                format_func=lambda x: {
                    "medical": "🏥 Medical Records & Clinical Trials", 
                    "legal": "⚖️ Indian Legal Contracts & Regulations", 
                    "resume": "📄 Resumes & Professional CV Credentials"
                }[x],
                label_visibility="collapsed"
            )

            st.divider()
            
            st.markdown("<p style='font-weight:600; font-size:1.1rem; margin-bottom:8px;'>2. Execution Parameters (5 Required)</p>", unsafe_allow_html=True)
            
            params_dict = {}
            detail_map = {"Concise": 1, "Standard": 2, "Comprehensive": 3}
            
            if selected_domain == "medical":
                c1, c2 = st.columns(2)
                params_dict["extract_dosages"] = c1.checkbox("1. Extract Dosages & Metrics", value=True)
                params_dict["map_pathology"] = c1.checkbox("2. Map Pathological History", value=True)
                params_dict["anonymize_pii"] = c2.checkbox("3. HIPAA Masking (Anonymize PII)", value=False)
                params_dict["ner_threshold"] = c2.slider("4. NER Confidence Threshold", 50, 99, 85)
                choice = st.select_slider("5. Summary Detail Generation", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"] = detail_map[choice]
            
            elif selected_domain == "legal":
                c1, c2 = st.columns(2)
                params_dict["isolate_signees"] = c1.checkbox("1. Isolate Signees & Parties", value=True)
                params_dict["scan_liabilities"] = c1.checkbox("2. Scan Financial Liabilities", value=True)
                params_dict["anonymize_pii"] = c2.checkbox("3. Redact Corporate PII", value=False)
                params_dict["risk_sensitivity"] = c2.slider("4. Risk Matrix Sensitivity", 1, 10, 7)
                choice = st.select_slider("5. Summary Detail Generation", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"] = detail_map[choice]
                
            elif selected_domain == "resume":
                c1, c2 = st.columns(2)
                params_dict["strict_skills"] = c1.checkbox("1. Strict Skill Matrix Matching", value=True)
                params_dict["parse_academic"] = c1.checkbox("2. Academic Timeline Parse", value=True)
                params_dict["anonymize_pii"] = c2.checkbox("3. Blind Screening (Anonymize Name)", value=False)
                params_dict["target_role"] = c2.text_input("4. Target Role Alignment (Optional)", placeholder="e.g. Lead AI Engineer")
                choice = st.select_slider("5. Profile Summary Detail", options=["Concise", "Standard", "Comprehensive"], value="Standard")
                params_dict["summary_detail"] = detail_map[choice]

            st.divider()

            st.markdown("<p style='font-weight:600; font-size:1.1rem; margin-bottom:8px;'>3. Upload Target File</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Document Source (PDF)", type=["pdf"], label_visibility="collapsed")
            
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Begin Deep Analysis Sequence", use_container_width=True, type="primary"):
                if not uploaded_file:
                    st.error("⚠️ Ingestion halted: Please supply a clean PDF file stream.")
                else:
                    with st.status("Initializing Ingestion Engines...", expanded=True) as status:
                        st.write("🧬 Binding deep-learning parameters to system cores...")
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"domain": selected_domain, "params": json.dumps(params_dict)}
                        
                        try:
                            response = requests.post(f"{BACKEND_URL}/process", files=files, data=data)
                            if response.status_code == 200:
                                st.session_state.processed_payload = response.json()
                                st.session_state.doc_id = st.session_state.processed_payload["document_id"]
                                st.session_state.active_domain = selected_domain
                                st.session_state.app_state = "dashboard"
                                status.update(label="Analysis Sequence Finalized.", state="complete", expanded=False)
                                time.sleep(0.3)
                                st.rerun()
                            else:
                                status.update(label="Pipeline Fault", state="error", expanded=True)
                                st.error(f"Backend Server Exception: {response.text}")
                        except Exception as e:
                            status.update(label="Network Timeout", state="error", expanded=True)
                            st.error(f"Failed to bridge connection parameters with backend: {str(e)}")

# ==========================================
# SCREEN 2: CLEAN DECONGESTED DASHBOARD
# ==========================================
elif st.session_state.app_state == "dashboard":
    payload = st.session_state.processed_payload
    domain = st.session_state.active_domain
    extracted_fields = payload.get("extracted_data", {})
    
    # --- MINIMALIST WORKSPACE SIDEBAR ---
    with st.sidebar:
        st.markdown("<h3 style='letter-spacing:-0.02em;'>📁 Session Meta</h3>", unsafe_allow_html=True)
        st.caption("Target File Stream")
        st.code(payload.get('filename', 'Source context'), language='text')
        st.caption("Active Extraction Domain")
        st.info(f"Target Cluster: {domain.upper()}")
        
        st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
        if st.button("🔌 Close Active Session", use_container_width=True, type="secondary"):
            reset_app()
            st.rerun()

    # --- TOP LINE SPACIOUS KPI CARDS ---
    st.markdown(f"<h1 style='letter-spacing: -0.04em; font-weight: 800; margin-bottom: 4px;'>{domain.capitalize()} Intelligence Control</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; font-size: 1.05rem; margin-bottom: 2rem;'>Decongested operational layout mapping document attributes onto isolated processing scopes.</p>", unsafe_allow_html=True)
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{domain.upper()}</div><div class="kpi-lbl">Core Pipeline Target</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        extracted_count = len(extracted_fields.keys()) if extracted_fields else 0
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{extracted_count} Keys</div><div class="kpi-lbl">Isolated Context Points</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{st.session_state.doc_id[-6:].upper()}</div><div class="kpi-lbl">Active Vector Hash ID</div></div>', unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # --- THREE TIER INTELLIGENCE TABS ---
    tab_summary, tab_analytics, tab_rag = st.tabs(["📋 Executive Summaries", "📊 Domain Deep-Dive Matrix", "💬 Citation Chat Sandbox"])
    
    # --- TAB 1: VERTICAL SUMMARIES ---
    with tab_summary:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        st.markdown("##### ⚡ Brief Capsule Insight")
        st.markdown(f"<div class='short-summary-box'>{payload.get('short_summary', 'No short insights processed.')}</div>", unsafe_allow_html=True)
            
        st.markdown("##### 📝 Comprehensive Engineering Breakdown")
        st.markdown(f"<div class='long-summary-box'>{payload.get('detailed_summary', 'No summary payload provided.')}</div>", unsafe_allow_html=True)

    # --- TAB 2: ADAPTIVE FEATURE MATRICES ---
    with tab_analytics:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Targeted Field-Extraction Resolution")
        
        if not extracted_fields:
            st.info("No field features found within this file block.")
        else:
            col_left, col_right = st.columns(2, gap="large")
            
            for idx, (field, content) in enumerate(extracted_fields.items()):
                target_col = col_left if idx % 2 == 0 else col_right
                with target_col:
                    # Determine styling classes depending on high-risk parameters
                    style_class = "domain-card"
                    if domain == "medical" and field in ["Diagnoses & Conditions", "Medications & Dosages"]:
                        style_class += " medical-alert"
                    elif domain == "legal" and field in ["Risks & Red Flags", "Financial Liabilities"]:
                        style_class += " legal-warning"
                    elif domain == "resume" and field in ["Target Role Alignment", "Certifications"]:
                        style_class += " resume-success"
                        
                    card_template = f"""
                    <div class="{style_class}">
                        <div class="card-title">📌 {field}</div>
                        <div class="card-body">{content}</div>
                    </div>
                    """
                    st.markdown(card_template, unsafe_allow_html=True)

    # --- TAB 3: CONTEXT-BOUNDED RAG WINDOW ---
    with tab_rag:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💬 Vector-Isolated Document Interrogation")
        st.caption("Queries execute purely against localized context vectors. Hallucinations are actively minimized via structural distance metrics.")
        
        chat_frame = st.container(height=480, border=True)
        
        with chat_frame:
            if not st.session_state.chat_log:
                st.markdown("<p style='text-align:center; color:#4b5563; margin-top:150px;'>🧠 Semantic index loaded. Query the tracking vector space natively below.</p>", unsafe_allow_html=True)
            else:
                for entry in st.session_state.chat_log:
                    with st.chat_message(entry["role"]):
                        st.markdown(entry["text"])
                        if entry.get("citations"):
                            for cite in entry["citations"]:
                                st.markdown(f"**Source Document Block:** `Page {cite['page']}`")
                                st.caption(f"_{cite['text_snippet']}_")
                                
        query_bar = st.chat_input("Ask a factual question regarding current document context strings...")
        if query_bar:
            st.session_state.chat_log.append({"role": "user", "text": query_bar})
            with chat_frame:
                with st.chat_message("user"):
                    st.markdown(query_bar)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Traversing multi-domain matrix..."):
                        req_body = {
                            "document_id": st.session_state.doc_id,
                            "domain": domain,
                            "question": query_bar
                        }
                        try:
                            res = requests.post(f"{BACKEND_URL}/query", json=req_body)
                            if res.status_code == 200:
                                ans_payload = res.json()
                                answer_text = ans_payload.get("answer", "Context parameter lookup failed.")
                                citations_list = ans_payload.get("citations", [])
                                
                                st.markdown(answer_text)
                                for cite in citations_list:
                                    st.markdown(f"**Source Document Block:** `Page {cite['page']}`")
                                    st.caption(f"_{cite['text_snippet']}_")
                                    
                                st.session_state.chat_log.append({
                                    "role": "assistant",
                                    "text": answer_text,
                                    "citations": citations_list
                                })
                            else:
                                st.error("The retrieval pipeline failed to collect matching vector nodes.")
                        except Exception as network_err:
                            st.error(f"Connection framework disrupted: {str(network_err)}")