import streamlit as st
import requests
import time

# Import shared configuration from same package
from .ui_config import API_BASE_URL


def check_api_health():
    """Check if the API is reachable"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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


def get_config():
    """Get all configuration parameters from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def settings_page():
    """Settings page for indexing control"""
    st.title("⚙️ Settings")

    # Display status section
    st.header("RAG System Availability")

    # Backend connection status
    is_connected = check_api_health()
    st.markdown("**Backend Status:**")
    if is_connected:
        st.markdown("🟢 CONNECTED")
    else:
        st.markdown("🔴 DISCONNECTED")

    st.markdown("---")

    # Check API health
    if not is_connected:
        st.error("Cannot connect to the backend API. Please ensure the server is running.")
        return

    # Get current indexing status
    status_data = get_indexing_status()

    # Display status section
    st.header("Indexing Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        status = status_data.get("status", "unknown")
        status_class = f"status-{status}"
        status_emoji = {
            "idle": "⚪",
            "running": "🟡",
            "completed": "🟢",
            "failed": "🔴"
        }.get(status, "⚫")

        st.markdown(f"**Current Status:**")
        st.markdown(f'<p class="{status_class}">{status_emoji} {status.upper()}</p>', unsafe_allow_html=True)

    with col2:
        if status_data.get("started_at"):
            st.markdown("**Started At:**")
            st.write(status_data["started_at"])

    with col3:
        if status_data.get("completed_at"):
            st.markdown("**Completed At:**")
            st.write(status_data["completed_at"])

    # Display message
    if status_data.get("message"):
        if status == "failed":
            st.error(f"{status_data['message']}")
        elif status == "completed":
            st.success(f"{status_data['message']}")
        else:
            st.info(f"{status_data['message']}")

    # Display results
    if status_data.get("results"):
        st.subheader("Indexing Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Dense Index (ChromaDB):**")
            dense_result = status_data["results"].get("dense", {})
            st.json(dense_result)

        with col2:
            st.markdown("**Sparse Index (BM25):**")
            sparse_result = status_data["results"].get("sparse", {})
            st.json(sparse_result)

    st.divider()

    # Indexing controls
    st.header("Trigger Indexing")

    is_running = status == "running"

    col1, col2 = st.columns(2)

    with col1:
        refresh_fixed = st.checkbox(
            "Refresh Fixed Wikipedia Pages",
            value=False,
            disabled=is_running,
            help="Re-scrape and re-index the fixed set of Wikipedia pages"
        )

    with col2:
        refresh_random = st.checkbox(
            "Refresh Random Wikipedia Pages",
            value=False,
            disabled=is_running,
            help="Fetch and index a new set of random Wikipedia pages"
        )

    limit = st.number_input(
        "Limit (leave empty or 0 for no limit)",
        min_value=0,
        value=0,
        disabled=is_running,
        help="Maximum number of Wikipedia pages to process (0 or None means no limit)"
    )

    # Convert 0 to None for API call
    if limit == 0:
        limit = None

    st.markdown("---")

    # Trigger button
    if is_running:
        st.button("Start Indexing", disabled=True, use_container_width=True)
        st.warning("Indexing is currently in progress. Please wait for it to complete.")
    else:
        if st.button("Start Indexing", use_container_width=True, type="primary"):
            with st.spinner("Triggering indexing process..."):
                result = trigger_indexing(refresh_fixed, refresh_random, limit)

                if result.get("status") == "accepted":
                    st.success("Indexing started successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to start indexing: {result.get('message', 'Unknown error')}")

    # Auto-refresh status when running
    if is_running:
        st.info("Auto-refreshing status every 5 seconds...")
        time.sleep(5)
        st.rerun()

    # Manual refresh button
    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.header("Configuration and Parameters")
    # List all the configurations and parameters used for indexing in a collapsible section for reference
    with st.expander("Current system configuration used for indexing and retrieval"):
        config_data = get_config()

        if config_data.get("status") == "error":
            st.error(f"Failed to fetch configuration: {config_data.get('message', 'Unknown error')}")
        elif config_data.get("status") == "success" and config_data.get("config"):
            config = config_data["config"]

            # Model Configuration
            st.subheader("Model Configuration")
            model_data = {
                "Parameter": ["Embedding Model", "Reranker Model", "LLM Model"],
                "Value": [
                    str(config.get("EMBEDDING_MODEL", "N/A")),
                    str(config.get("RERANKER_MODEL", "N/A")),
                    str(config.get("LLM_MODEL", "N/A"))
                ]
            }
            st.table(model_data)

            # LLM Parameters
            st.subheader("LLM Parameters")
            llm_config = config.get("LLM_CONFIG", {})
            llm_data = {
                "Parameter": ["Max New Tokens", "Temperature", "Top P"],
                "Value": [
                    str(llm_config.get("max_new_tokens", "N/A")),
                    str(llm_config.get("temperature", "N/A")),
                    str(llm_config.get("top_p", "N/A"))
                ]
            }
            st.table(llm_data)

            # Retrieval Configuration
            st.subheader("Retrieval Configuration")
            retrieval_config = config.get("RETRIEVAL_CONFIG", {})
            retrieval_data = {
                "Parameter": ["Dense Top K", "Sparse Top K", "RRF Top K", "Reranker Top K", "Reranking Enabled"],
                "Value": [
                    str(retrieval_config.get("dense_top_k", "N/A")),
                    str(retrieval_config.get("sparse_top_k", "N/A")),
                    str(retrieval_config.get("rrf_top_k", "N/A")),
                    str(retrieval_config.get("reranker_top_k", "N/A")),
                    "Yes" if config.get("IS_RERANKING_ENABLED") else "No"
                ]
            }
            st.table(retrieval_data)

            # Data Collection
            st.subheader("Data Collection")
            data_config = config.get("DATA_COLLECTION_CONFIG", {})
            data_collection_data = {
                "Parameter": ["Fixed Sample Size", "Random Sample Size"],
                "Value": [
                    str(data_config.get("fixed_sample_size", "N/A")),
                    str(data_config.get("random_sample_size", "N/A"))
                ]
            }
            st.table(data_collection_data)

            # Complete Configuration JSON
            st.subheader("Complete Configuration (JSON)")
            st.json(config)
        else:
            st.warning("No configuration data available")
