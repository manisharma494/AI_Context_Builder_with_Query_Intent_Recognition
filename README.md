# AI Context Builder with Query Intent Recognition

This project is a robust Python app that simulates a Retrieval-Augmented Generation (RAG) pipeline for Philippine History. It loads a PDF, chunks and embeds the content using local sentence-transformers and FAISS, retrieves relevant chunks for a user query, builds a context, and generates an answer using an LLM (OpenAI GPT-3.5 or later).

## Features
- Loads and chunks a Philippine history PDF (only once; index is persisted)
- Embeds chunks into a FAISS vector index using local HuggingFace sentence-transformers
- Accepts user queries and retrieves top-N most relevant chunks
- Builds a context string from retrieved chunks
- Sends the context and question to an LLM (OpenAI GPT-3.5-turbo)
- Prints the LLM's response clearly to the console
- **Beautiful, classical, and professional Streamlit UI**
- Comprehensive unit tests for all major flows and edge cases
- GitHub Actions CI for automated testing

## UI Design Principles
- **Whitespace**: Generous spacing for a clean, uncluttered look
- **Grayscale-first, strong contrast**: Designed for clarity, with color as an accent
- **Refined typography**: Clear type scale, consistent font, and letter spacing
- **Soft shadows and depth**: Cards and buttons have subtle, realistic shadows
- **Interesting backgrounds**: Gentle gradients for a modern, elegant feel
- **Micro-interactions and animation**: Fade-in effects, button hover transitions, and interactive chunk cards
- **Section headers**: Clear, visually distinct sections for query, retrieved chunks, and LLM response

## Setup
1. Clone this repository.
2. Place the `philippine_history.pdf` file in the `data/` folder (download from NCCA).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install faiss-cpu langchain-huggingface sentence-transformers
   ```
4. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env to set TEAMIFIED_OPENAI_API_KEY
   ```

## Running the App
```bash
python run.py
```
You will be prompted for a user query. The script will print the retrieved chunks and the LLM's answer.
- The FAISS index and chunks are saved to disk after the first run for fast subsequent queries.

## Web API (FastAPI)
You can run the backend API server with:
```bash
uvicorn app_api:app --reload
```
- The API will be available at `http://localhost:8000`.
- Query endpoint: `POST /query` with JSON `{ "question": "...", "top_n": 5 }`
- Health endpoint: `GET /health`

## Web UI (Streamlit)
You can run the Streamlit UI with:
```bash
streamlit run app_ui.py
```
- The UI will connect to the FastAPI backend at `http://localhost:8000` by default.
- Enter your question, adjust the number of chunks, and view the answer and retrieved context interactively.
- The UI is designed for a beautiful, classical, and professional user experience.

## LLM and Embedding Choice
- **Embeddings:** Local, using `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`.
- **LLM:** OpenAI GPT-3.5-turbo (or later) via the OpenAI API.
- The OpenAI API key is read from the `TEAMIFIED_OPENAI_API_KEY` environment variable.

## Testing
- The app includes a comprehensive test suite using `pytest` and `unittest.mock`.
- To run all tests locally:
  ```bash
  PYTHONPATH=. pytest tests
  ```
- To run API tests:
  ```bash
  PYTHONPATH=. pytest tests/test_api.py
  ```
- Tests cover all sample queries, edge cases, and error handling.

## Continuous Integration (CI)
- GitHub Actions workflow is included in `.github/workflows/python-app.yml`.
- All tests are run automatically on every push and pull request.

## Example Usage
```
User Query: When did the EDSA People Power Revolution happen?

Retrieved Chunks:
- "The EDSA People Power Revolution occurred in February 1986..."
- "It led to the ousting of President Ferdinand Marcos..."

LLM Response:
"The EDSA People Power Revolution happened in February 1986 and marked the end of Marcos' dictatorship in the Philippines."
```

## Notes
- The script expects the PDF at `data/philippine_history.pdf`.
- All dependencies are listed in `requirements.txt`.
- The app is robustly tested and CI-enabled for maintainability and scalability.

## Bonus: Add More Tests
- The test suite is easy to extend with more queries and edge cases.

---

**Enjoy exploring Philippine history with a modern, beautiful RAG pipeline!** 