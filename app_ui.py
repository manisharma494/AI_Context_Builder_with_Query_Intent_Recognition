import streamlit as st
import requests
import time
import base64

st.set_page_config(page_title="Philippine History RAG App", layout="centered")

# --- Custom CSS for beautiful, classical, and modern UI ---
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #f8f6f2 0%, #e6e2d3 100%);
            padding-bottom: 2em;
        }}
        .main-title {{
            font-family: 'Georgia', serif;
            color: #222;
            font-size: 3rem;
            text-align: center;
            margin-top: 2em;
            margin-bottom: 0.2em;
            letter-spacing: 0.04em;
        }}
        .subtitle {{
            font-family: 'Georgia', serif;
            color: #555;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 2.5em;
            letter-spacing: 0.02em;
        }}
        .section-title {{
            font-family: 'Georgia', serif;
            color: #333;
            font-size: 1.3rem;
            margin-top: 2em;
            margin-bottom: 0.5em;
            letter-spacing: 0.03em;
        }}
        .retrieved-chunk {{
            background: #fff;
            border-radius: 14px;
            box-shadow: 0 4px 24px #e6e2d3, 0 1.5px 4px #c19a6b33;
            border: 1.5px solid #e6e2d3;
            padding: 1.1em 1.5em;
            margin-bottom: 1.2em;
            font-family: 'Georgia', serif;
            color: #3d2c1e;
            font-size: 1.05rem;
            transition: box-shadow 0.2s, border 0.2s;
            animation: fadeIn 0.7s;
        }}
        .retrieved-chunk:hover {{
            box-shadow: 0 8px 32px #c19a6b55, 0 2px 8px #e6e2d3;
            border: 1.5px solid #c19a6b;
        }}
        .llm-response {{
            background: linear-gradient(90deg, #e6f2ff 0%, #f8f6f2 100%);
            border-radius: 14px;
            box-shadow: 0 4px 24px #c9e6ff, 0 1.5px 4px #4682b433;
            border: 1.5px solid #c9e6ff;
            padding: 1.3em 2em;
            font-size: 1.15rem;
            font-family: 'Georgia', serif;
            color: #1a2634;
            margin-top: 1.5em;
            margin-bottom: 2em;
            transition: box-shadow 0.2s, border 0.2s;
            animation: fadeIn 1.2s;
        }}
        .llm-response:hover {{
            box-shadow: 0 8px 32px #4682b455, 0 2px 8px #c9e6ff;
            border: 1.5px solid #4682b4;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #222 0%, #c19a6b 100%);
            color: #fff;
            font-family: 'Georgia', serif;
            font-size: 1.1rem;
            border-radius: 8px;
            border: none;
            padding: 0.7em 2.5em;
            margin-top: 1em;
            margin-bottom: 1.5em;
            box-shadow: 0 2px 8px #e6e2d3;
            transition: 0.2s;
            letter-spacing: 0.04em;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #c19a6b 0%, #222 100%);
            color: #fffbe6;
        }}
        .stTextInput>div>div>input {{
            background: #fff;
            border-radius: 8px;
            border: 1.5px solid #e6e2d3;
            font-size: 1.1rem;
            font-family: 'Georgia', serif;
            padding: 0.7em 1em;
        }}
        .stSlider>div {{
            padding-top: 0.5em;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg()

st.markdown('<div class="main-title">Philippine History RAG App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about Philippine history and get concise, context-based answers.</div>', unsafe_allow_html=True)

API_URL = "http://localhost:8000/query"
HEALTH_URL = "http://localhost:8000/health"

# Check backend status
try:
    health_resp = requests.get(HEALTH_URL, timeout=2)
    backend_ok = health_resp.status_code == 200
except Exception:
    backend_ok = False

if not backend_ok:
    st.error("⚠️ The backend API is not running. Please start it with: `uvicorn app_api:app --reload`.")
    st.stop()

with st.form("query_form"):
    st.markdown('<div class="section-title">Query</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("Enter your question:", key="user_query")
    with col2:
        top_n = st.slider("Chunks", 1, 10, 5, key="top_n")
    submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted and user_query.strip():
    with st.spinner("Retrieving answer..."):
        time.sleep(0.5)  # Subtle animation effect
        try:
            resp = requests.post(API_URL, json={"question": user_query, "top_n": top_n})
            if resp.status_code == 200:
                data = resp.json()
                st.markdown('<div class="section-title">Retrieved Chunks</div>', unsafe_allow_html=True)
                for chunk in data["retrieved_chunks"]:
                    st.markdown(f'<div class="retrieved-chunk">{chunk[:300]}...</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">LLM Response</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="llm-response">{data["llm_response"]}</div>', unsafe_allow_html=True)
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Enter a question and click 'Ask' to get started.") 