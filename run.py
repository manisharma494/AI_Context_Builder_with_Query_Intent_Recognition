import os
import sys
import fitz  # PyMuPDF
import pickle
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

PDF_PATH = 'data/philippine_history.pdf'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_N = 5  # Increased for more context
INDEX_PATH = 'data/faiss_index'
CHUNKS_PATH = 'data/chunks.pkl'


def load_pdf_chunks(pdf_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_or_load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("Loading FAISS index and chunks from disk...")
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        return vectorstore, chunks
    else:
        print("Index not found. Building FAISS index from PDF...")
        if not os.path.exists(PDF_PATH):
            print(f"PDF file not found at {PDF_PATH}")
            sys.exit(1)
        chunks = load_pdf_chunks(PDF_PATH)
        print(f"Loaded {len(chunks)} chunks.")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        return vectorstore, chunks


def retrieve_chunks(vectorstore, query: str, top_n: int = TOP_N):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_n)
    return [(doc.page_content, score) for doc, score in docs_and_scores]


def build_context(chunks: List[str]) -> str:
    return '\n'.join([f'- "{chunk}"' for chunk in chunks])


def ask_llm(context: str, question: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("TEAMIFIED_OPENAI_API_KEY"))
    prompt = (
        "You are a helpful Philippine history expert. "
        "Using only the provided context, answer the user's question in a single, clear, and accurate sentence. "
        "Your answer must include both the date and the significance of the event, and should be concise and self-contained. "
        "Do not add any information that is not in the context. "
        "If the context is insufficient, say 'Not enough information in the context.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    if hasattr(response, 'content'):
        return response.content
    return str(response)


def main():
    vectorstore, _ = build_or_load_index()
    print("User Query:", end=' ')
    user_query = input().strip()
    print("\nRetrieving relevant chunks...")
    retrieved = retrieve_chunks(vectorstore, user_query, TOP_N)
    retrieved_chunks = [chunk for chunk, _ in retrieved]
    print("\nRetrieved Chunks:")
    for chunk in retrieved_chunks:
        # Print only the first sentence or up to 80 chars, as in the sample
        first_sentence = chunk.split(". ")[0] + ("..." if "." in chunk else "")
        print(f'- "{first_sentence.strip()}"')
    context = build_context(retrieved_chunks)
    print("\nLLM Response:")
    response = ask_llm(context, user_query)
    print(f'"{response.strip()}"')

if __name__ == "__main__":
    main() 