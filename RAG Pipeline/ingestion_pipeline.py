import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader # load documents
from langchain_text_splitters import CharacterTextSplitter # document into chunks
from langchain_huggingface import HuggingFaceEmbeddings # chunk into embeddings
from langchain_chroma import Chroma # store embeddings (Chroma is a vector database)
from dotenv import load_dotenv # load environment variables
load_dotenv()


def load_documents(docs_path="docs"):
    """Load documents from a directory"""
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )

    documents = loader.load()
    if len(documents)==0:
        raise FileNotFoundError("No documents found in the directory")

    for i,doc in enumerate(documents):
        print(f"Document {i+1}: ")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-"*50)

    return documents

# use CharacterTextSplitter to split documents into chunks
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into chunks"""
    print("Splitting documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i,chunk in enumerate(chunks[:5]):
            print(f"---Chunk {i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Content preview: {chunk.page_content[:100]}...")
            print(f"Metadata: {chunk.metadata}")
            print("-"*50)
        
        if len(chunks) > 5:
            print(f"\n....and {len(chunks)-5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create a vector store from chunks"""
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create ChromaDB vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"} # for similarity search
    )
    print("--- Finish creating embeddings and storing in ChromaDB---")

    print(f"Vector store created at {persist_directory}")

    return vectorstore


def main():
    print("Main function")

    # load documents
    documents = load_documents(docs_path="docs")

    # split documents into chunks
    chunks = split_documents(documents)

    # chunks into embeddings and store in vector database
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()