import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from run import build_or_load_index, retrieve_chunks, build_context, ask_llm, TOP_N
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None

@app.on_event("startup")
def load_index_on_startup():
    global vectorstore
    vectorstore, _ = build_or_load_index()

class QueryRequest(BaseModel):
    question: str
    top_n: int = TOP_N

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_api(req: QueryRequest):
    if vectorstore is None:
        # Return a 400 error with a clear message and the expected keys
        raise HTTPException(status_code=400, detail="Index not loaded.")
    retrieved = retrieve_chunks(vectorstore, req.question, req.top_n)
    retrieved_chunks = [chunk for chunk, _ in retrieved]
    context = build_context(retrieved_chunks)
    llm_response = ask_llm(context, req.question)
    return {
        "question": req.question,
        "retrieved_chunks": retrieved_chunks,
        "llm_response": llm_response.strip()
    } 