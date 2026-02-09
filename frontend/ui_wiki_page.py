import streamlit as st
import requests

# Import shared configuration from same package
from .ui_config import API_BASE_URL


def check_api_health():
    """Check if the API is reachable"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_fixed_wiki_pages():
    """Get list of fixed wiki pages from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/wiki/fixed", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_random_wiki_pages():
    """Get list of random wiki pages from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/wiki/random", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def wiki_pages_page():
    """Wikipedia pages browser"""
    st.title("📚 Wikipedia Pages")

    # Check API health
    if not check_api_health():
        st.error("Cannot connect to the backend API. Please ensure the server is running.")
        return

    # Create tabs for fixed and random pages
    tab1, tab2 = st.tabs(["Fixed Pages", "Random Pages"])

    with tab1:
        fixed_pages_data = get_fixed_wiki_pages()

        if fixed_pages_data.get("status") == "error":
            st.error(f"Error: {fixed_pages_data.get('message')}")
        elif fixed_pages_data.get("status") == "success":
            pages = fixed_pages_data.get("data", {}).get("pages", [])
            count = fixed_pages_data.get("count", 0)

            if count > 0:
                st.caption(f"{count} pages indexed")

                # Group pages by category
                categories = {}
                for page in pages:
                    cat = page.get('category', 'Uncategorized')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(page)

                # Display by category
                for category, cat_pages in sorted(categories.items()):
                    with st.expander(f"**{category.replace('_', ' ')}** ({len(cat_pages)} pages)"):
                        for page in sorted(cat_pages, key=lambda x: x.get('page_title', '')):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                url = page.get('url', '')
                                title = page.get('page_title', 'Untitled')
                                if url:
                                    st.markdown(f"[{title}]({url})")
                                else:
                                    st.markdown(title)

                            with col2:
                                st.caption(f"ID: {page.get('page_id', 'N/A')}")
            else:
                st.info("No pages found")

    with tab2:
        random_pages_data = get_random_wiki_pages()

        if random_pages_data.get("status") == "error":
            st.error(f"Error: {random_pages_data.get('message')}")
        elif random_pages_data.get("status") == "success":
            pages = random_pages_data.get("data", {}).get("pages", [])
            count = random_pages_data.get("count", 0)

            if count > 0:
                st.caption(f"{count} pages indexed")

                # Search
                search = st.text_input("🔍 Search", key="search_random", placeholder="Filter by title...")

                # Filter
                filtered = [p for p in pages if search.lower() in p.get('page_title', '').lower()] if search else pages

                st.caption(f"Showing {len(filtered)} pages")

                # Display
                for page in sorted(filtered, key=lambda x: x.get('page_title', '')):
                    with st.expander(f"**{page.get('page_title', 'Untitled')}**"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            url = page.get('url', '')
                            title = page.get('page_title', 'Untitled')
                            if url:
                                st.markdown(f"[{title}]({url})")
                            else:
                                st.markdown(title)

                        with col2:
                            st.caption(f"ID: {page.get('page_id', 'N/A')}")
            else:
                st.info("No pages found")
