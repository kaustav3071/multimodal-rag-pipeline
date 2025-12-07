from dotenv import load_dotenv
load_dotenv()
import os
import re
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "db/chroma_db"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_model
)   

# Create a AI model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct",
    temperature=0.3
)

# store our conversation history
conversation_history = []

def ask_question(user_question):
    print(f"---User Question: {user_question}---")

    # Step 1: Make the question clear using conversation history
    if conversation_history:
        history_text = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in conversation_history
        ])
        
        # Ask AI to make the question clear
        messages = [
            SystemMessage(content=f"""You are a question reformulation assistant. Your job is to take a follow-up question and rewrite it to be standalone and clear.

Conversation History:
{history_text}

Instructions:
- Look at the conversation history above
- Rewrite the current question to include relevant context from the history
- Make it a complete, standalone question
- Output ONLY the reformulated question, nothing else
- Do not add explanations or prefixes

Examples:
- If history mentions "Tesla" and question is "Where is it located?", output: "Where is Tesla located?"
- If history mentions "NVIDIA" and question is "Who founded it?", output: "Who founded NVIDIA?"
- If history mentions "SpaceX rockets" and question is "When were they first launched?", output: "When were SpaceX rockets first launched?"
"""),
            HumanMessage(content=f"{user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        
        # Clean up any model-generated tags or artifacts
        search_question = re.sub(r'<s>|</s>|\[/?OUT\]|\[/?INST\]', '', search_question).strip()
        search_question = re.sub(r'^[-\s]+|[-\s]+$', '', search_question).strip()
        
        # Fallback to original question if reformulation is empty
        if not search_question:
            print("[Warning: Reformulation returned empty, using original question]")
            search_question = user_question
        else:
            print(f"---Reformulated Search Question: {search_question}---")
    
    else:
        search_question = user_question
    
    # Step 2: Search for relevant documents
    retrieved_docs = db.as_retriever(search_kwargs={"k": 3})
    docs = retrieved_docs.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")
    for i,doc in enumerate(docs):
        # Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}: {preview}...")

    # Step 3 : Create final prompt
    context_text = "\n\n".join(doc.page_content for doc in docs)
    combined_input = f"""You are a helpful assistant. Answer the following question using ONLY the information from the provided context.

Context:
{context_text}

Question: {user_question}

Instructions:
- Provide a clear, direct answer
- Use only information from the context above
- Do not add external knowledge
- If the context doesn't contain the answer, say "I cannot find this information in the provided documents"

Answer:"""

    # Step 4 : Generate response
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content
    
    # Check if answer is empty
    if not answer or not answer.strip():
        answer = "[No answer generated - the model returned an empty response]"
    
    # Add to conversation history
    conversation_history.append({
        "question": user_question,
        "answer": answer
    })

    print("\n---Generated Response---\n")
    print("Content only:")
    print(answer)
    print()  # Add blank line for readability


def start_chat():
    print("\n---Chat with AI---\n")
    while True:
        user_question = input("User: ")
        if user_question.lower() == "exit":
            print("\n---Chat ended---\n")
            break
        ask_question(user_question) 

if __name__ == "__main__":
    start_chat()
