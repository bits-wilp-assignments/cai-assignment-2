import streamlit as st


def about_page():
    """About page with project and team information"""
    st.title("📋 About")

    # Project Information
    st.header("Hybrid RAG Chatbot")
    # Display logo in main page
    st.markdown("""
    #### BITs Pilani - WILP
    **Conversational AI - Assignment 2**
    """)
    st.markdown("""
    This is an advanced **Retrieval-Augmented Generation (RAG)** system that combines multiple retrieval
    strategies to provide accurate and contextual responses to user queries.
    The system integrates:
    - **Dense retrieval** - ChromaDB with embeddings for semantic search.
    - **Sparse retrieval** - BM25 for keyword-based search.
    - **Re-ranking** - Cross-encoder model for improved answer quality.
    """)

    # Team Information
    st.header("Team Members - Group 56")
    st.markdown("")

    # Custom CSS for team member cards
    st.markdown("""
    <style>
        .team-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .team-name {
            color: #333333;
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .team-id {
            color: #666666;
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }
    </style>
    """, unsafe_allow_html=True)

    # Team members in a clean grid layout
    col1, col2 = st.columns(2)

    team_members = [
        ("Abhishek Kumar Tiwari", "2024AA05192"),
        ("Krishanu Chakraborty", "2024AA05193"),
        ("Viswanadha Pavan Kumar", "2024AA05197"),
        ("B Vinod Kumar", "2024AA05832"),
        ("K Abhinav", "2024AB05168")
    ]

    for idx, (name, roll_id) in enumerate(team_members):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
            <div class="team-card">
                <div class="team-name">{name}</div>
                <div class="team-id">ID: {roll_id}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.caption("Built with ❤️ for the WILP CAI course project.")
