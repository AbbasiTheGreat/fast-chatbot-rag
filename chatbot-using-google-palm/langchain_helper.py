from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from dotenv import load_dotenv


# load_dotenv()
import os

llm = GooglePalm(google_api_key='AIzaSyD_ge0fV4dqE-JjWmwvVBfcOAaWUWsm6Jw', temperature=0.5)



# Creating Embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path="faiss_index"    
    

def get_qa_chain():
    # Load the Database from local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    
    # Create a retriever for querying the database
    retriever = vectordb.as_retriever(score_threshold=0.7)
    

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT})
    
    return chain
    
    
    
    
if __name__ == "__main__":
    chain = get_qa_chain()


