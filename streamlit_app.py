import streamlit as st
import requests
import json

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")
st.title("🔄 Self-Correcting RAG System")
st.caption("Ask a question. The system will self-correct if it detects hallucinations.")

API_URL = "http://localhost:8000"

# Sidebar: Info
with st.sidebar:
    st.header("📊 System Info")
    st.markdown("""
    **How it works:**
    1. 🔍 RAG retrieves relevant context
    2. 🤖 LLM generates answer
    3. 🧠 LLM-as-Judge evaluates answer
    4. 🔄 If hallucinated → Auto-correct (max 3 tries)
    5. ✅ Returns final answer
    """)
    st.divider()
    st.caption("Powered by Groq LLM + LangGraph")

# Main chat area
st.header("💬 Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander("📊 Evaluation Details"):
                st.json(msg["metadata"])

if prompt := st.chat_input("Ask something..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Call API
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking with self-correction..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display answer
                    st.markdown(data["answer"])
                    
                    # Show correction info
                    if data["attempts"] > 1:
                        st.info(f"🔄 Self-corrected after {data['attempts']} attempts")
                    
                    if data["is_valid"]:
                        st.success(f"✅ Valid answer (Confidence: {data['confidence']:.2f})")
                    else:
                        st.warning(f"⚠️ Answer may contain hallucinations (Confidence: {data['confidence']:.2f})")
                    
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": data["answer"],
                        "metadata": {
                            "attempts": data["attempts"],
                            "is_valid": data["is_valid"],
                            "confidence": data["confidence"],
                            "hallucinated": data["hallucinated"]
                        }
                    })
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")