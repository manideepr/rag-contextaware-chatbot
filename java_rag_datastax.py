import os
import openai
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key
JAVA_PROJECT_PATH = os.getenv("JAVA_PROJECT_PATH")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")  # From your Astra DB credentials
ASTRA_DB_ID = "java_rag"  # Found in your connect bundle
ASTRA_DB_REGION = "us-east2"  # Example region
ASTRA_DB_KEYSPACE = "default_keyspace"
COLLECTION_NAME = "java_rag_coll"

# === SETUP ===
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
embedding_model_name = "text-embedding-3-small"
CHUNK_TOKEN_LIMIT = 800
tokenizer = tiktoken.encoding_for_model(embedding_model_name)

# === Set up LangChain's AstraDB vector store ===
embedding_model = OpenAIEmbeddings(model=embedding_model_name)

vectorstore = AstraDB(
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint="https://8f57de35-dbfb-4b47-ac10-16b2dc5b7e35-us-east-2.apps.astra.datastax.com",
    namespace=ASTRA_DB_KEYSPACE
)

# === Helper Functions ===

def get_java_files(root_dir):
    java_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def chunk_text(text, max_tokens):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        if len(tokenizer.encode(" ".join(current_chunk))) > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# === Process and Upload to Astra ===
java_files = get_java_files(JAVA_PROJECT_PATH)
documents = []
metadatas = []

for filepath in java_files:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    chunks = chunk_text(code, CHUNK_TOKEN_LIMIT)
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "filename": filepath,
            "chunk_id": i
        })

print(f"📦 Preparing to embed and upload {len(documents)} code chunks...")

# Upload to Astra (this will call OpenAI for embeddings internally)
vectorstore.add_texts(texts=documents, metadatas=metadatas)

print("✅ Embeddings uploaded successfully to Astra DB.")
