from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path="faiss_index"

def create_vector_db():
    # Load CSV FIle
    loader = CSVLoader(file_path='fastnuces_faqs.csv', source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

if __name__ == "__main__":
    create_vector_db()
