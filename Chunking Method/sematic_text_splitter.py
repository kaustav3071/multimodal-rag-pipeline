from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Semantic Text Splitter = groups text into chunks based on semantic similarity
#(group by meaning)
semantic_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    breakpoint_threshold_type="percentile", # can be "percentile" or "similarity"
    breakpoint_threshold_amount=70 # 70th percentile of semantic similarity
)

chunks = semantic_splitter.split_text(text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: ({len(chunk)} characters)")
    print(chunk)
    print("\n")

print("\n")
