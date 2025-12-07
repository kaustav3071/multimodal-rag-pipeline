import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
from retrieval_pipeline import relevant_docs

query = "What was NVIDIA's first graphics accelerator called ?"

context = "\n\n".join(
    doc.page_content for doc in relevant_docs
)

# Combined the query and the relevant documents contents
combined_input = f"""Based on the following context, answer the question:
Context: {context}
Question: {query}

Please provide a clear, helpful answer using only these documents. Don't include any other information and don't include any citations.
"""

# Create a AI model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct",
    temperature=0.3
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n---Generated Response---\n")
print("Content only:")
print(result.content)

