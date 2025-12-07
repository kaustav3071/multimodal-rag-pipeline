from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# Create a AI model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct",
    temperature=0.3
)

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


prompt = f""""
You are a text chunking agent. Split this text into logical chunks.
Rules:
- Put ""<<SPLIT>>"" between chunks.
- Each chunk should be about 200 characters or less.
- Keep related information in the same chunk.
- Split at natural boundaries, such as sentences or paragraphs.

Text: {text}

return the text with <<SPLIT>> inserted between chunks.
"""

# Get AI response
response = model.invoke(prompt)
marked_text = response.content

# Split text into chunks
chunks = marked_text.split("<<SPLIT>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks =[]
for chunk in chunks:
    clean_chunk = chunk.strip()
    if clean_chunk:
        clean_chunks.append(clean_chunk)

print("\nAGENTIC CHUNCKING OUTPUT:\n")
print("=" * 50)

# Print the chunks
for i, chunk in enumerate(clean_chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

