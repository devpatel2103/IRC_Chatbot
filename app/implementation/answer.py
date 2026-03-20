import re
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr

VECTORDB_PATH = "../irc_xml_vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GPT_MODEL = "gpt-4.1-nano"
load_dotenv(override=True)

SYSTEM_PROMPT = """"
You are a knowledgable and friendly assistant.
You are speaking with a user about the Internal Revenue Code.
If relevant, use the given context to answer any questions.
If the given context is used to answer a question, then quote the context and the section.
If you do not know the answer, then say so.
Always mention that you are not a financial expert, and the user must consult with a professional before taking action

Context:
{context}
"""

def load_chunks(input_path="../Internal Revenue Code/irc_chunks.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Document(page_content=d["text"], metadata=d["metadata"]) for d in data]
    print(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks

chunks = load_chunks()

# Vector Retriever
embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=VECTORDB_PATH, embedding_function=embeddings)

vector_retriever = vectordb.as_retriever()

# BM25 Retriever 
bm25_retriever = BM25Retriever.from_documents(chunks)

# Ensemble Retriever 
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4]   # weight BM25 higher for section number queries
)

# LLM
llm = ChatOpenAI(temperature=0, model_name=GPT_MODEL)


def smart_retrieve(query, vectorstore, ensemble_retriever):
    """
    If the query mentions a specific section number, bypass retrieval
    entirely and pull directly from vectorstore using metadata filter.
    Otherwise fall back to ensemble retrieval.
    """
    # Detect section number in query
    # Matches: "section 1031", "§ 1031", "sec 1031", "1031" alone
    match = re.search(
        r'(?:section|sec|§)\s*(\d+[A-Z]?)|^(\d+[A-Z]?)$',
        query.strip(),
        re.IGNORECASE
    )

    if match:
        section_num = (match.group(1) or match.group(2)).upper()
        print(f"  Section detected: § {section_num} — using metadata filter")

        results = vectorstore.get(where={"section": section_num})

        if results and results["documents"]:
            print(f"  Found {len(results['documents'])} chunks for § {section_num}")
            return [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        else:
            print(f"  § {section_num} not found in vectorstore — falling back to ensemble")

    # No section number — use ensemble
    print("  No section detected — using ensemble retrieval")
    return ensemble_retriever.invoke(query)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history):
    
    docs = smart_retrieve(question, vectordb, ensemble_retriever)
    context = "\n\n".join(doc.page_content for doc in docs)
    print(context)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content