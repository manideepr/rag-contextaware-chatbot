import os
import openai
import tiktoken
import weaviate
from weaviate.auth import Auth
import weaviate.client as weaviate_client
from dotenv import load_dotenv

load_dotenv

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Parameters
CHUNK_TOKEN_LIMIT = 800  # To stay within model limits
MODEL = "text-embedding-3-small"

# Tokenizer for counting tokens
tokenizer = tiktoken.encoding_for_model(MODEL)

def get_java_files(root_dir):
    java_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def chunk_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        token_count = len(tokenizer.encode(" ".join(current_chunk)))
        if token_count > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def embed_chunks(chunks, metadata_prefix=""):
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = openai.embeddings.create(
            input=chunk,
            model=MODEL
        )
        embedding = response.data[0].embedding
        embeddings.append({
            "embedding": embedding,
            "chunk_id": i,
            "text": chunk,
            "metadata": f"{metadata_prefix}_chunk_{i}"
        })
    return embeddings

# Main processing
def process_java_project(project_path):
    all_embeddings = []
    java_files = get_java_files(project_path)

    for file_path in java_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            chunks = chunk_text(code, CHUNK_TOKEN_LIMIT)
            file_embeddings = embed_chunks(chunks, metadata_prefix=file_path)
            all_embeddings.extend(file_embeddings)

    return all_embeddings

openai.api_key=os.getenv("OPENAI_API_KEY")
# Example usage
project_path ="E:\\JavaProofOfConcepts\\developer-joyofenergy-java"
embeddings = process_java_project(project_path)


# Add this:
print(f"✅ Total embeddings created: {len(embeddings)}")
print("🧾 Sample embedding metadata and text:")
for item in embeddings[:2]:  # Show first 2 chunks
    print("—" * 40)
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Metadata: {item['metadata']}")
    print(f"Text (first 100 chars): {item['text'][:100]}")


# 1. Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://wbvk1etrushrnyejqqpsg.c0.us-west3.gcp.weaviate.cloud",  # from Weaviate Console
    auth_credentials=Auth.api_key(api_key="nzBLG2dPsA4KCnSmKMcdvwr2PAnNhZNjOGvi"),
)

# 2. Create a schema if not already present
class_name = "CodeChunk"

if not client.schema.contains({"class": class_name}):
    client.schema.create_class({
        "class": class_name,
        "vectorizer": "none",  # since you're using OpenAI embeddings
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "chunk_id", "dataType": ["int"]},
        ]
    })

# 3. Push each embedding as an object
for item in embeddings:
    obj = {
        "text": item["text"],
        "filename": item["metadata"].split("_chunk_")[0],
        "chunk_id": item["chunk_id"]
    }
    client.data_object.create(
        data_object=obj,
        class_name=class_name,
        vector=item["embedding"]
    )

print("✅ Embeddings uploaded to Weaviate.")
