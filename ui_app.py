import streamlit as st
import requests
from typing import Generator

# Import page modules from ui_components package
from streamlit_ui.ui_wiki_page import wiki_pages_page
from streamlit_ui.ui_settings_page import settings_page
from streamlit_ui.ui_about_page import about_page
from streamlit_ui.ui_config import API_BASE_URL

# Page configuration
st.set_page_config(
    page_title="BITs Assistant AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide deploy button */
    .stAppDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1.2rem !important;
        border-radius: 0.8rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    }

    /* User message styling */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f5f5f5 !important;
        border-left: 4px solid #2196F3 !important;
    }

    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: #f5f5f5 !important;
        border-left: 4px solid #2196F3 !important;
    }

    /* Assistant message styling */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: #f5f5f5 !important;
        border-left: 4px solid #4CAF50 !important;
    }

    /* Avatar styling */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }

    /* Chat input styling */
    [data-testid="stChatInput"] {
        border-radius: 0.8rem !important;
    }

    /* Status indicators styling */
    .status-running {
        color: #ff9800;
        font-weight: bold;
    }
    .status-completed {
        color: #4caf50;
        font-weight: bold;
    }
    .status-failed {
        color: #f44336;
        font-weight: bold;
    }
    .status-idle {
        color: #9e9e9e;
        font-weight: bold;
    }

    /* Markdown content styling in messages */
    [data-testid="stChatMessage"] code {
        background-color: rgba(0,0,0,0.1) !important;
        padding: 0.2em 0.4em !important;
        border-radius: 3px !important;
    }

    [data-testid="stChatMessage"] pre {
        background-color: rgba(0,0,0,0.05) !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is reachable"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_streaming_response(question: str) -> Generator[str, None, None]:
    """Get streaming response from the inference endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference",
            json={"question": question},
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    chunk = line_str[6:]  # Remove 'data: ' prefix
                    if chunk.startswith('[ERROR]'):
                        yield f"\n\n{chunk}"
                        break
                    else:
                        yield chunk
    except requests.exceptions.RequestException as e:
        yield f"\n\nError: {str(e)}"


def trigger_indexing(refresh_fixed: bool, refresh_random: bool, limit: int = None):
    """Trigger the indexing process"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/index/trigger",
            json={
                "refresh_fixed": refresh_fixed,
                "refresh_random": refresh_random,
                "limit": limit
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_indexing_status():
    """Get the current indexing status"""
    try:
        response = requests.get(f"{API_BASE_URL}/index/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def chat_page():
    """Main chat interface"""
    st.title("BITs Assistant AI")

    # Check API health
    if not check_api_health():
        st.error("Cannot connect to the backend API. Please ensure the server is running.")
        st.info("Run: `python hybrid-rag-app.py` or `uvicorn hybrid-rag-app:app --reload`")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history container
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Fixed input at bottom
    question = st.chat_input("Type your question here...", key="chat_input_main")

    # Process question
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

        # Get and display bot response with streaming
        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                # Show thinking spinner initially
                with response_placeholder:
                    with st.spinner("Thinking..."):
                        # Get first chunk to start streaming
                        full_response = ""
                        stream_gen = get_streaming_response(question)

                        # Try to get the first chunk
                        try:
                            first_chunk = next(stream_gen)
                            full_response = first_chunk
                        except StopIteration:
                            pass

                # Clear spinner and show streaming content
                if full_response:
                    response_placeholder.markdown(full_response)

                    # Continue streaming remaining chunks
                    for chunk in stream_gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response)

                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Rerun to update chat history
        st.rerun()


def main():
    """Main application"""
    # Initialize page in session state if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Chat"

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")

        # Get the current index based on session state
        page_options = ["Chat", "Wiki Pages", "Settings", "About"]
        if st.session_state.current_page in page_options:
            current_index = page_options.index(st.session_state.current_page)
        else:
            current_index = 0

        page = st.radio(
            "Go to:",
            page_options,
            index=current_index,
            key="page_selector",
            label_visibility="collapsed"
        )

        # Update session state immediately
        st.session_state.current_page = page

    # Route to appropriate page - ONLY call the selected page function
    # This ensures chat_input is never created when on Settings or Wiki Pages
    if page == "Wiki Pages":
        st.session_state.current_page = "Wiki Pages"
        # Hide chat input
        st.markdown("""
        <style>
            [data-testid="stChatInput"] {
                display: none !important;
            }
        </style>
        """, unsafe_allow_html=True)
        wiki_pages_page()
    elif page == "Settings":
        st.session_state.current_page = "Settings"
        # Hide chat input with CSS and JavaScript
        st.markdown("""
        <style>
            [data-testid="stChatInput"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                opacity: 0 !important;
            }
        </style>
        <script>
            // Remove chat input element if it exists
            setTimeout(function() {
                const chatInputs = window.parent.document.querySelectorAll('[data-testid="stChatInput"]');
                chatInputs.forEach(input => input.remove());
            }, 100);
        </script>
        """, unsafe_allow_html=True)
        settings_page()
    elif page == "About":
        st.session_state.current_page = "About"
        # Hide chat input
        st.markdown("""
        <style>
            [data-testid="stChatInput"] {
                display: none !important;
            }
        </style>
        """, unsafe_allow_html=True)
        about_page()
    else:  # Default to Chat
        st.session_state.current_page = "Chat"
        chat_page()


if __name__ == "__main__":
    main()
