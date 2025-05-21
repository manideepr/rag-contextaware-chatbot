from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    print("Hello from root!")
    return {"message": "Hello!"}

@app.post("/chat")
async def chat_endpoint():
    print("Hello from chat!")
    return {"answer": "It works!"}
