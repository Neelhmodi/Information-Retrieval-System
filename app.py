import streamlit as st
import time
from src.helpers import get_pdf_text, get_text_chunks, create_vector_store, create_conversational_chain

def user_input(user_question):
    response = st.session_state.conversational_chain({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User:** {message['content']}")  # User messages
        else:
            st.write(f"**AI:** {message['content']}")  # AI responses

def main():
    st.set_page_config(page_title="Information Retrieval System", page_icon=":mag:")
    st.header("Information Retrieval System")

    user_question = st.text_input("Ask a question about the PDF content:")

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_file = st.file_uploader("Upload a PDF file", accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Simulate PDF processing time
                time.sleep(2)
                # Here you would add your PDF processing logic
                pdf_text = get_pdf_text(pdf_file)
                text_chunks = get_text_chunks(pdf_text)
                vector_store = create_vector_store(text_chunks)
                st.session_state.conversational_chain = create_conversational_chain(vector_store)
                st.success("PDF processed successfully!")

if __name__ == "__main__":
    main()