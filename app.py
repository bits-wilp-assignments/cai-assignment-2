# import streamlit as st
from src.service.inference import rag_inference
from src.service.indexing import triggr_indexing
import streamlit as st

# Streamlit interface
st.title("Hybrid RAG System")





# if __name__ == "__main__":

    # triggr_indexing(is_refresh_fixed=False, is_refresh_random=False)

    # while True:
    #     question = input("Enter your question: ")
    #     rag_inference(question)

# 1. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Accept user input
if prompt := st.chat_input("Ask me about the documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Display assistant response in chat message container
    with st.chat_message("assistant"):
        # This is where your RAG chain streaming goes (see next step)
        pass