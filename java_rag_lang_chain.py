import os
import weaviate
from weaviate.auth import AuthApiKey
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate as LCWeaviate
from langchain.chains import RetrievalQA
import os
import weaviate
from weaviate.auth import AuthApiKey
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

load_dotenv()

# ==== SETUP ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # Name used during embedding

# Set OpenAI key for LangChain
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ==== Connect to Weaviate ====
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
   # headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}  # Needed if using Weaviate's hybrid/gen AI modules
)



# ==== Set up LangChain Vector Store ====
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # ⚠️ Must match your document embeddings!

vectorstore = LCWeaviate(
    client=client,
    index_name=COLLECTION_NAME,
    text_key="text",
    embedding=embedding_model
)

# ==== Set up RetrievalQA Chain ====
llm = ChatOpenAI(model="gpt-4")  # or "gpt-3.5-turbo"

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 300}),
    return_source_documents=True
)

# ==== Run a Query ====
# query = "Explaing CostCalculatorController class?"
# result = qa_chain(query)

# query_embedding = embedding_model.embed_query("Explaing CostCalculatorController class?")

# # Vector search (uses nearVector)
# docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)

# for doc in docs:
#     print(doc.metadata.get("filename"))
#     print(doc.page_content[:300])

# ==== FORMAT CONTEXT ====
def format_docs(docs, max_chars=1000):
    return "\n\n".join(
        f"File: {doc.metadata.get('filename', '[unknown]')}\n{doc.page_content[:max_chars]}"
        for doc in docs
    )

# ==== CUSTOM PROMPT ====
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Java software assistant. Use the provided code snippets to answer clearly."),
    ("human", "Here is relevant code context:\n\n{context}\n\nQuestion: {question}")
])

# ==== LLM SETUP ====
llm = ChatOpenAI(model="gpt-4")

# ==== BUILD RAG CHAIN ====
# retriever = vectorstore.similarity_search_by_vector(query_embedding, k=30)

# rag_chain = (
#     {
#         "context": retriever | format_docs,
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )
output_parser = StrOutputParser()

# === CHAT LOOP ===
print("💬 Chat with your Java codebase (Ctrl+C or 'exit' to quit)\n")
# while True:
#     query = input("👤 You: ").strip()
#     if query.lower() in {"exit", "quit"}:
#         break

#     try:
#         # Embed the query and fetch similar chunks
#         query_vector = embedding_model.embed_query(query)
#         docs = vectorstore.similarity_search_by_vector(query_vector, k=3)
#         context = format_docs(docs, max_chars=5000)

#         # Build and run the prompt
#         chain = prompt | llm | output_parser
#         response = chain.invoke({"context": context, "question": query})

#         print("\n🤖 GPT:\n" + response)
#         print("-" * 60)

#     except Exception as e:
#         print(f"❌ Error: {e}")

# --- FastAPI code starts here ---

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    print("Hello from root!")
    return {"message": "Hello!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    print(f"👤 You: {request.question}")
    query = request.question.strip()
    if query.lower() in {"exit", "quit"}:
        return {"answer": ""}
    try:
        # Embed the query and fetch similar chunks
        query_vector = embedding_model.embed_query(query)
        docs = vectorstore.similarity_search_by_vector(query_vector, k=3)
        context = format_docs(docs, max_chars=5000)
        # Build and run the prompt
        chain = prompt | llm | output_parser
        response = chain.invoke({"context": context, "question": query})

        print("\n🤖 GPT:\n" + response)
        print("-" * 60)
        return response
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # try:
    #     result = qa_chain({"question": request.question})
    #     return {"answer": result["answer"]}
    # except Exception as e:
    #     return {"error": str(e)}