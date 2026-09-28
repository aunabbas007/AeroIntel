# AeroIntel ✈️

AeroIntel is an aviation intelligence system that uses Retrieval-Augmented Generation (RAG) to answer questions about airlines and aircraft using a curated aviation knowledge base.

The project combines semantic search with a Large Language Model (LLM) to retrieve relevant information before generating an answer.

---

## 🚀 Overview

AeroIntel was built to understand and implement a RAG pipeline from scratch before introducing higher-level frameworks such as LangChain.

The system currently contains information about:

- Airlines
- Aircraft
- Airline fleets
- Aircraft specifications

Users can ask questions such as:

- Which aircraft does Emirates operate?
- What aircraft are in the Air India fleet?
- Compare the Airbus A350-900 and Boeing 787-9.
- Which airlines operate the Airbus A380?

---

## 🧠 Architecture

```text
                KNOWLEDGE BASE
                      │
                      ▼
                 Document Loader
                      │
                      ▼
                    Chunking
                      │
                      ▼
                  Embeddings
                      │
                      ▼
                Pinecone Vector DB
                      │
                      │
User Question ────────┘
       │
       ▼
 Query Embedding
       │
       ▼
 Semantic Retrieval
       │
       ▼
 Relevant Context
       │
       ▼
      Gemini
       │
       ▼
    Final Answer
```
🔄 RAG Pipeline
1. Document Loading

AeroIntel loads aviation documents stored as .txt files.

data/
├── airlines/
└── aircrafts/

Each document is classified using metadata such as:

Document type
Entity ID
Display name
Source
2. Chunking

Documents are divided into smaller pieces before generating embeddings.

The current implementation uses paragraph-aware chunking with:

Chunk size: 300 words
Overlap: 50 words

Chunking allows the retrieval system to work with smaller, more relevant pieces of information.

3. Embeddings

AeroIntel uses:

all-MiniLM-L6-v2

from Sentence Transformers.

Each chunk is converted into a 384-dimensional vector.

Text
 ↓
Embedding Model
 ↓
384-dimensional vector
4. Vector Database

The embeddings are stored in:

Pinecone

AeroIntel uses cosine similarity to retrieve semantically relevant information.

The vector metadata contains information such as:

text
source
type
entity_id
display_name
5. Semantic Retrieval

When the user asks a question, the question is converted into an embedding.

Pinecone then searches for the most semantically similar chunks.

The system supports:

Top-k retrieval
Similarity score thresholds
Metadata filtering
Entity filtering
6. Context Construction

The retrieved chunks are combined into a context that is passed to the LLM.

Retrieved Documents
        ↓
Context Builder
        ↓
Prompt + Context
7. LLM Generation

AeroIntel uses Google's Gemini model to generate the final response.

The model is instructed to answer using the retrieved context rather than relying on outside knowledge.

🛡️ Grounded Responses

AeroIntel includes a simple grounding mechanism.

If relevant information cannot be retrieved, the system responds:

I don't have enough information in the knowledge base to answer that question.

This prevents the application from confidently answering questions that are outside its knowledge base.

🗂️ Project Structure
AeroIntel/
│
├── data/
│   ├── airlines/
│   └── aircrafts/
│
├── source/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── context.py
│   └── generator.py
│
├── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
source/

Contains the individual components of the RAG pipeline.

File	Purpose
loader.py	Loads aviation documents
chunking.py	Splits documents into chunks
embeddings.py	Generates text embeddings
vector_store.py	Handles Pinecone operations and retrieval
context.py	Builds context from retrieved chunks
generator.py	Generates answers using Gemini
app.py

The main application entry point.

It connects the retrieval and generation components into one end-to-end pipeline.

⚙️ Installation
1. Clone the repository
git clone https://github.com/aunabbas007/AeroIntel.git
cd AeroIntel
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key

Do not commit .env to GitHub.

▶️ Running AeroIntel

Start the application with:

python app.py

You will then be prompted for a question:

Ask AeroIntel:

Example:

Ask AeroIntel: Which aircraft does Emirates operate?

Example output:

AeroIntel:

Based on the provided context, Emirates operates the following aircraft:

- Airbus A380
- Airbus A350
- Boeing 777
- 
🧪 Example Queries

Airline query
Which aircraft does Emirates operate?
Aircraft query
What is the Airbus A350-900?
Comparison
Compare the Airbus A350-900 and Boeing 787-9.
Knowledge boundary
Who won the 2026 World Cup?

If the information is not available in the knowledge base, AeroIntel refuses to answer rather than using external knowledge.

🧰 Technologies

Python
Sentence Transformers
Hugging Face
Pinecone
Google Gemini
NumPy
python-dotenv

🎯 Learning Objectives

This project was built to understand the internal components of a RAG system before using higher-level frameworks.

The project covers:

Document ingestion
Text chunking
Embeddings
Semantic similarity
Vector databases
Metadata filtering
Retrieval
Context construction
Prompt construction
LLM generation
Grounded generation
Basic hallucination control
🔮 Future Development

AeroIntel can be extended with:

LangChain
Advanced retrieval strategies
Reranking
Query rewriting
Source citations
RAG evaluation
LangGraph
Agentic workflows
MCP
FastAPI
Docker
Cloud deployment

These features are intentionally outside the current implementation because the goal of this version is to understand the fundamentals of RAG.

👨‍💻 Author

Aun Abbas
