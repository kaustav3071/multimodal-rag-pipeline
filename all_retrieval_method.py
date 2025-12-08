from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

persist_directory = "db/chroma_db"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much Microsoft pay to acquire GitHub?"
print(f"Query: {query}")

# ---------------------------------------
# METHOD 1: Basic similarity search
# ---------------------------------------

print("=== METHOD 1: Basic similarity search ===")
retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke(query)


for i,doc in enumerate(docs):
    print(f"\n\n----------------\n\n")
    print(f"Document {i+1}")
    print(doc.page_content)
print("-"*50)

# ---------------------------------------
# METHOD 2: Similarity with Score Threshold
# Only returns documents above a certain similarity score
# ---------------------------------------

print("\n=== METHOD 2: Similarity with Score Threshold ===")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.45  # Only return docs with similarity >= 0.45
    }
)

docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents (threshold: 0.45):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)

# ---------------------------------------
# METHOD 3: Maximum Marginal Relevance (MMR)
# Balances relevance and diversity - avoids redundant results
# ---------------------------------------

print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # Final number of docs
        "fetch_k": 10,  # Initial pool to select from
        "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
    }
)

docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("=" * 60)
print("Done! Try different queries or parameters to see the differences.")
