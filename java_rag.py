import os
import openai
import tiktoken
import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv

load_dotenv()
#from weaviate.collections.classes.config import Property, DataType

# ==== CONFIGURATION ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # Optional, only if auth is enabled
JAVA_PROJECT_PATH = os.getenv("JAVA_PROJECT_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ==== OPENAI SETUP ====
openai.api_key = OPENAI_API_KEY
EMBED_MODEL = "text-embedding-3-small"
CHUNK_TOKEN_LIMIT = 800
tokenizer = tiktoken.encoding_for_model(EMBED_MODEL)

# ==== WEAVIATE CONNECTION ====
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
   # headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}  # Needed if using Weaviate's hybrid/gen AI modules
)

# ==== SCHEMA SETUP ====
existing_classes = [cls["class"] for cls in client.schema.get()["classes"]]

if COLLECTION_NAME not in existing_classes:
    client.schema.create_class({
        "class": COLLECTION_NAME,
        "vectorizer": "none",  # We're using external OpenAI vectors
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "chunk_id", "dataType": ["int"]}
        ]
    })

    print(f"✅ Class '{COLLECTION_NAME}' created.")
else:
    print(f"ℹ️ Class '{COLLECTION_NAME}' already exists.")

#collection = client.collections.get(COLLECTION_NAME)

# ==== JAVA FILE LOADING ====
def get_java_files(root_dir):
    java_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.endswith(".java")
    ]
    
    print(f"📄 Found {len(java_files)} Java files.")
    return java_files

# ==== TEXT CHUNKING ====
def chunk_text(text, max_tokens):
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if len(tokenizer.encode(" ".join(current))) > max_tokens:
            chunks.append(" ".join(current[:-1]))
            current = [word]

    if current:
        chunks.append(" ".join(current))
    return chunks

# ==== EMBEDDING FUNCTION ====
def embed_text(text):
    return openai.embeddings.create(
        input=text,
        model=EMBED_MODEL
    ).data[0].embedding

# ==== PROCESS AND PUSH ====
java_files = get_java_files(JAVA_PROJECT_PATH)
total_chunks = 0

for path in java_files:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    chunks = chunk_text(code, CHUNK_TOKEN_LIMIT)
    print(f"📦 Preparing to embed and upload {len(chunks)} code chunks from {path}...")

    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk)
            client.data_object.create(
                data_object={
                    "text": chunk,
                    "filename": path,
                    "chunk_id": i
                },
                class_name=COLLECTION_NAME,
                vector=embedding
            )
            total_chunks += 1
        except Exception as e:
            print(f"❌ Error embedding {path} chunk {i}: {e}")

print(f"✅ Successfully embedded and stored {total_chunks} chunks.")

# ==== OPTIONAL SEARCH FUNCTION ====
def search_code(query, k=3):
    # Get the embedding of the query
    query_embedding = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search in Weaviate using near_vector
    response = client.query.get(
        class_name=COLLECTION_NAME,
        properties=["text", "filename"]
    ).with_near_vector({
        "vector": query_embedding
    }).with_limit(k).do()

    # Print results
    print(f"\n🔍 Top {k} results for: '{query}'\n")
    for obj in response["data"]["Get"][COLLECTION_NAME]:
        print("📂 File:", obj.get("filename", "[no filename]"))
        print(obj.get("text", "")[:300])
        print("-" * 40)

# Uncomment to test search
search_code("Explain joyofEnergy project?")
