import streamlit as st
from langchain_helper import get_qa_chain

st.title("Chatbot")

btn = st.button("FAST-NUCES")

if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    
    st.header("Answer: ")
    st.write(response["result"])
