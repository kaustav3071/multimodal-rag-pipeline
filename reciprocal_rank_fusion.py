from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import os

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.3
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ---------------------------------------
# Step 1: Generate Multiple Query Variations
# ---------------------------------------

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)

# ---------------------------------------
# Step 2: Search with Each Query Variation & Store Results
# ---------------------------------------

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)

print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")

# ---------------------------------------
# Step 3: Calculate Reciprocal Rank Fusion
# ---------------------------------------

def reciprocal_rank_fusion(chunk_list, k=60, verbose=True):

    if verbose:
        print("\n"+"="*60)
        print("\n=== Reciprocal Rank Fusion ===")
        print("\n"+"="*60)
        print(f"\nUsing k={k} for RRF calculation\n")
        print("Calculating Reciprocal Rank Fusion...")

    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}           # Will store: {chunk_content: actual_chunk_object}

    # For verbose output – track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1

    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_list, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")

        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content

            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk

            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

            # Add to RRF score
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(f"Chunk {chunk_id} (position {position}): {position_score}")
                print(f"Preview: {chunk.page_content[:150]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        all_unique_chunks.items(), 
        key=lambda x: rrf_scores[x[0]],  # Sort by RRF score, where x[0] is chunk_content
        reverse=True
    )

    # Return just the chunk objects (not the tuples)
    return [chunk_obj for chunk_content, chunk_obj in sorted_chunks]

fused_result = reciprocal_rank_fusion(all_retrieval_results,k=60,verbose=True)

# ---------------------------------------
# Step 4: Display Fused Results
# ---------------------------------------

print("\n"+"="*60)
print("\n=== Fused Results ===")
print("\n"+"="*60)

for i, chunk in enumerate(fused_result, 1):
    print(f"\nChunk {i}:")
    print(chunk.page_content[:150] + "...")

print("\n"+"="*60)
print("Fused Results Complete!")
