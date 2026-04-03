# ============================================================
# 03_rag_agent.py
# ============================================================
# RAG Agent -- ChromaDB + LangChain RecursiveCharacterTextSplitter
# Architecture:
#   Documents
#     -> RecursiveCharacterTextSplitter (smart chunking)
#     -> Embedding Model (NOT the LLM!)
#     -> ChromaDB (vector store + metadata)
#     -> Similarity Search (query time)
#     -> LLM generates answer with retrieved context
# ============================================================

import os
import shutil
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import chat, embed


# ===========================================================
# Part 1: Custom Embedding Function for ChromaDB
# ===========================================================
# ChromaDB needs an EmbeddingFunction object.
# We wrap our config.embed() so ChromaDB can call it
# automatically during add() and query().
# ===========================================================


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction):
    """
    Wraps any OpenAI-compatible embedding API (LM Studio / OpenAI / Ollama)
    into a ChromaDB-compatible EmbeddingFunction.

    This bridges config.embed() <-> ChromaDB's interface.
    """

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB calls this with a list of strings.
        We call our config.embed() which talks to the Embedding API.
        """
        # embed() from config.py supports both single string and list
        results = []
        for text in input:
            vec = embed(text)
            results.append(vec)
        return results


# ===========================================================
# Part 2: Smart Document Chunking
# ===========================================================
# Instead of crude fixed-size character slicing, we use
# LangChain's RecursiveCharacterTextSplitter which respects
# natural text boundaries (paragraphs > sentences > words).
# ===========================================================


def create_text_splitter(chunk_size=300, chunk_overlap=50):
    """
    Create a smart text splitter.

    RecursiveCharacterTextSplitter tries separators in order:
      1. "\n\n"  (paragraph breaks)
      2. "\n"    (line breaks)
      3. ". "    (sentence ends)
      4. ", "    (clause breaks)
      5. " "     (word breaks)
      6. ""      (character-level, last resort)

    This keeps semantically related text together as much as possible.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(documents, metadatas=None, chunk_size=300, chunk_overlap=50):
    """
    Split a list of documents into chunks, preserving metadata.

    Returns:
        chunks:    list of text strings
        metas:     list of metadata dicts (each includes 'source' and 'chunk_index')
        ids:       list of unique ID strings for ChromaDB
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)

    all_chunks = []
    all_metas = []
    all_ids = []
    doc_counter = 0

    for i, doc_text in enumerate(documents):
        meta = metadatas[i] if metadatas else {}
        chunks = splitter.split_text(doc_text)

        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append(
                {
                    **meta,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                }
            )
            all_ids.append(f"doc{i}_chunk{j}_{doc_counter}")
            doc_counter += 1

    return all_chunks, all_metas, all_ids


# ===========================================================
# Part 3: ChromaDB Vector Store Manager
# ===========================================================


class ChromaVectorStore:
    """
    Manages a ChromaDB collection with custom embedding function.

    Supports:
      - In-memory (ephemeral) or persistent storage
      - Automatic embedding via OpenAI-compatible API
      - Metadata filtering
      - Smart document chunking
    """

    def __init__(self, collection_name="rag_knowledge_base", persist_directory=None):
        """
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Path to persist data. None = in-memory only.
        """
        # Create ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"[ChromaDB] Persistent mode: {persist_directory}")
        else:
            self.client = chromadb.EphemeralClient()
            print("[ChromaDB] Ephemeral mode (in-memory)")

        # Create embedding function
        self.embedding_fn = OpenAICompatibleEmbeddingFunction()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        print(
            f"[ChromaDB] Collection '{collection_name}' ready. "
            f"Existing documents: {self.collection.count()}"
        )

    def add_documents(
        self, texts, metadatas=None, ids=None, chunk_size=300, chunk_overlap=50
    ):
        """
        Chunk and add documents to the collection.
        ChromaDB will automatically call our embedding function.
        """
        chunks, metas, auto_ids = chunk_documents(
            texts, metadatas, chunk_size, chunk_overlap
        )

        final_ids = ids or auto_ids

        # ChromaDB handles embedding automatically via embedding_function
        self.collection.add(
            documents=chunks,
            metadatas=metas,
            ids=final_ids,
        )

        print(
            f"[ChromaDB] Added {len(chunks)} chunks from {len(texts)} documents. "
            f"Total: {self.collection.count()}"
        )

    def search(self, query, top_k=3, where=None):
        """
        Search for relevant documents.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            where: Optional metadata filter, e.g. {"source": "rag-tutorial"}

        Returns:
            List of dicts with 'text', 'metadata', 'distance' keys.
        """
        query_params = {
            "query_texts": [query],
            "n_results": min(top_k, self.collection.count()),
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        # Flatten ChromaDB's nested response format
        output = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                output.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                        "distance": (
                            results["distances"][0][i] if results["distances"] else 0
                        ),
                    }
                )
        return output

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection.name)
        print(f"[ChromaDB] Collection deleted.")


# ===========================================================
# Part 4: RAG Agent
# ===========================================================


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent.

    Flow:
      1. User asks a question.
      2. Question is embedded (via ChromaDB's embedding function).
      3. ChromaDB returns top-K similar document chunks.
      4. Chunks are assembled into a context prompt.
      5. LLM generates an answer grounded in the retrieved context.
    """

    def __init__(self, vector_store, top_k=3):
        self.vector_store = vector_store
        self.top_k = top_k
        self.system_prompt = (
            "You are an intelligent Q&A assistant backed by a knowledge base.\n\n"
            "Rules:\n"
            "1. Answer based ONLY on the [Reference Materials] provided below.\n"
            "2. If the materials do not contain relevant information, clearly say:\n"
            "   'No relevant information found in the knowledge base.'\n"
            "3. Do NOT fabricate information beyond what is in the references.\n"
            "4. Cite source numbers, e.g. [1], [2], when referencing materials.\n"
            "5. Be concise and accurate.\n"
            "6. If multiple sources support the answer, synthesize them."
        )

    def _build_context(self, query, where=None):
        """Retrieve relevant chunks and build context string."""
        results = self.vector_store.search(query, top_k=self.top_k, where=where)

        if not results:
            return "No relevant materials found.", results

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            # ChromaDB returns distance (lower = more similar for cosine)
            similarity = 1 - r["distance"]  # Convert distance to similarity
            context_parts.append(
                f"[{i}] (source: {source}, similarity: {similarity:.3f})\n"
                f"{r['text']}"
            )

        return "\n\n".join(context_parts), results

    def ask(self, question, where=None):
        """
        Main RAG pipeline.

        Args:
            question: User's question.
            where: Optional metadata filter for ChromaDB.
        """
        print(f"\n{'='*60}")
        print(f"[Question] {question}")

        # Step 1: Retrieve (embedding is handled by ChromaDB automatically)
        context, results = self._build_context(question, where=where)
        print(f"[Retrieved] {len(results)} chunks")
        for r in results:
            sim = 1 - r["distance"]
            print(
                f"  - similarity={sim:.3f} | source={r['metadata'].get('source', '?')} "
                f"| {r['text'][:60]}..."
            )

        # Step 2: Augment prompt with context
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"[Reference Materials]\n{context}\n\n"
                    f"[User Question]\n{question}"
                ),
            },
        ]

        # Step 3: Generate answer (using LLM, NOT embedding model!)
        response = chat(messages, temperature=0.3)
        print(f"\n[Answer]\n{response.content}")
        return response.content

    def ask_with_history(self, question, chat_history=None, where=None):
        """
        RAG with conversation history for follow-up questions.
        """
        print(f"\n{'='*60}")
        print(f"[Follow-up Question] {question}")

        context, results = self._build_context(question, where=where)
        print(f"[Retrieved] {len(results)} chunks")

        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        if chat_history:
            for msg in chat_history:
                messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"[Reference Materials]\n{context}\n\n"
                    f"[User Question]\n{question}"
                ),
            }
        )

        response = chat(messages, temperature=0.3)
        print(f"\n[Answer]\n{response.content}")
        return response.content


# ===========================================================
# Part 5: Run -- Full Demo
# ===========================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Prepare knowledge base documents
    #    (In real projects, load from files/databases/APIs)
    # --------------------------------------------------

    documents = [
        # Document 1: LM Studio
        (
            "LM Studio is a desktop application that allows users to run "
            "large language models locally on their own hardware.\n\n"
            "Key features of LM Studio:\n"
            "- Supports GGUF format models from Hugging Face.\n"
            "- Provides an OpenAI-compatible API server.\n"
            "- Users can access the local model API via http://localhost:1234.\n"
            "- Supports both CPU and GPU inference.\n"
            "- Built-in chat interface for testing models.\n\n"
            "LM Studio makes it easy to experiment with open-source LLMs "
            "without needing cloud services or API keys."
        ),
        # Document 2: RAG
        (
            "RAG (Retrieval-Augmented Generation) is a technique that combines "
            "information retrieval with text generation.\n\n"
            "How RAG works:\n"
            "1. Documents are split into chunks and embedded into vectors.\n"
            "2. Vectors are stored in a vector database.\n"
            "3. When a user asks a question, the question is also embedded.\n"
            "4. Similar document chunks are retrieved via vector similarity search.\n"
            "5. Retrieved chunks are passed as context to the LLM.\n"
            "6. The LLM generates an answer grounded in the retrieved context.\n\n"
            "Benefits of RAG:\n"
            "- Reduces LLM hallucination by grounding responses in real data.\n"
            "- Enables LLMs to answer questions about private or recent data.\n"
            "- More cost-effective than fine-tuning for domain-specific knowledge.\n"
            "- Knowledge can be updated by simply updating the document store."
        ),
        # Document 3: Vector Databases
        (
            "Vector databases are specialized databases designed for storing "
            "and retrieving high-dimensional vector embeddings.\n\n"
            "Popular vector databases:\n"
            "- ChromaDB: Open-source, lightweight, great for prototyping. "
            "Supports in-memory and persistent storage.\n"
            "- Milvus: Open-source, highly scalable, supports billions of vectors.\n"
            "- Pinecone: Fully managed cloud service, easy to use but proprietary.\n"
            "- Qdrant: Open-source, written in Rust, high performance.\n"
            "- FAISS: Facebook's library for efficient similarity search.\n\n"
            "Vector databases use Approximate Nearest Neighbor (ANN) algorithms "
            "like HNSW (Hierarchical Navigable Small World) for fast retrieval. "
            "They are essential components in RAG pipelines and semantic search."
        ),
        # Document 4: Function Calling
        (
            "Function Calling is a mechanism that enables LLMs to invoke "
            "external tools and APIs.\n\n"
            "How Function Calling works:\n"
            "1. Developer defines tool schemas (name, description, parameters).\n"
            "2. Schemas are sent to the LLM along with the user's message.\n"
            "3. The LLM decides which tool to call and what arguments to pass.\n"
            "4. The application executes the tool and returns results to the LLM.\n"
            "5. The LLM generates a final response incorporating the tool results.\n\n"
            "Modern frameworks like LangChain provide @tool decorators that "
            "automatically generate JSON schemas from Python function signatures "
            "and docstrings, eliminating the need for manual schema writing."
        ),
        # Document 5: Multi-Agent Systems
        (
            "Multi-Agent systems consist of multiple specialized AI agents "
            "that collaborate to solve complex tasks.\n\n"
            "Common orchestration patterns:\n"
            "- Router: Directs tasks to the most suitable expert agent.\n"
            "- Sequential Pipeline: Agents process in order, each building "
            "on the previous output (e.g., Researcher -> Coder -> Writer).\n"
            "- Parallel Fan-out: Multiple agents work simultaneously, "
            "results are aggregated.\n"
            "- Hierarchical: A manager agent decomposes tasks and delegates "
            "to worker agents.\n"
            "- Debate: Multiple agents argue different perspectives, "
            "a judge synthesizes the conclusion.\n\n"
            "Each agent typically has its own system prompt, tools, and "
            "area of expertise. They coordinate through message passing."
        ),
    ]

    metadatas = [
        {"source": "lm-studio-docs", "category": "tools"},
        {"source": "rag-tutorial", "category": "techniques"},
        {"source": "vector-db-guide", "category": "infrastructure"},
        {"source": "function-calling-guide", "category": "techniques"},
        {"source": "multi-agent-paper", "category": "architecture"},
    ]

    # --------------------------------------------------
    # 2. Create ChromaDB vector store and index documents
    # --------------------------------------------------

    # Clean up previous runs (optional, for demo reproducibility)
    persist_dir = "./chromadb"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    store = ChromaVectorStore(
        collection_name="agent_knowledge_base",
        persist_directory=persist_dir,  # Set to None for in-memory
    )

    # Smart chunking + automatic embedding + storage
    store.add_documents(
        texts=documents,
        metadatas=metadatas,
        chunk_size=200,  # Each chunk ~200 chars
        chunk_overlap=30,  # 30 chars overlap for context continuity
    )

    # --------------------------------------------------
    # 3. Create RAG Agent and test
    # --------------------------------------------------

    agent = RAGAgent(store, top_k=3)

    # Test 1: General question
    agent.ask("What is RAG and what are its benefits?")

    # Test 2: Specific factual question
    agent.ask("What is the default port for LM Studio's local API?")

    # Test 3: Comparison question
    agent.ask("What are the popular vector databases and how do they compare?")

    # Test 4: Question about techniques
    agent.ask("How does Function Calling work with LLMs?")

    # Test 5: Filtered search (only search in specific sources)
    agent.ask(
        "What orchestration patterns exist for multi-agent systems?",
        where={"category": "architecture"},
    )

    # Test 6: Question NOT in knowledge base
    agent.ask("What should I eat for lunch today?")

    # --------------------------------------------------
    # 4. Multi-turn conversation with history
    # --------------------------------------------------

    print("\n" + "#" * 60)
    print("# Multi-turn Conversation Demo")
    print("#" * 60)

    history = []

    # Turn 1
    q1 = "What is RAG?"
    a1 = agent.ask_with_history(q1, chat_history=history)
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content": a1})

    # Turn 2 (follow-up)
    q2 = "Which vector databases can I use to build a RAG system?"
    a2 = agent.ask_with_history(q2, chat_history=history)
    history.append({"role": "user", "content": q2})
    history.append({"role": "assistant", "content": a2})

    # Turn 3 (follow-up referencing previous answers)
    q3 = "Among those you mentioned, which one is best for prototyping?"
    a3 = agent.ask_with_history(q3, chat_history=history)

    # --------------------------------------------------
    # 5. Show ChromaDB stats
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("[ChromaDB Stats]")
    print(f"  Collection: {store.collection.name}")
    print(f"  Total documents: {store.collection.count()}")
    print(f"  Persist directory: {persist_dir}")
    print("=" * 60)

    # Cleanup (comment out if you want to keep the data)
    # store.delete_collection()
    # shutil.rmtree(persist_dir)
