from re import search
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os

persist_directory = "db/chroma_db"

# load embeddings and vector store
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# Search for relevant documents
query = "What was NVIDIA's first graphics accelerator called ?"

retriever = db.as_retriever(search_kwargs={"k": 5}) # retrieve top 5 relevant documents which are most similar to the query

# retrieve top 5 relevant documents which are most similar to the query
relevant_docs = retriever.invoke(query) 

def main():
    """Run the retrieval pipeline as a standalone script"""
    print(f"User Query : {query}")
    print("-"*50)
    
    for i,doc in enumerate(relevant_docs,1):
        print(f"Document {i}:\n{doc.page_content}\n")

if __name__ == "__main__":
    main()

