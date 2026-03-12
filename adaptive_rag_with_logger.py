# this code is adaptive logger with logger
# i didnt used s3 or langsmith ,langstudio or any kind(may be imported but didnt used it)
# used only Structured Logging- Python logging + JSON
# didint used -> LLM Tracing LangSmith ,
# didnt used Full Observability LangFuse or langstudio or Arize Phoenix


# 0 LOGGER SETUP
import logging
import json
from datetime import datetime, timezone
from langchain_core.callbacks.base import BaseCallbackHandler


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "node": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)


def get_logger(node_name: str) -> logging.Logger:
    logger = logging.getLogger(node_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("rag_pipeline.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger
    

class LoggerCallback(BaseCallbackHandler):
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger = get_logger("LLM")
        logger.info("Prompt sent to LLM", extra={"extra": {
            "prompt": prompts[0][:300]   # first 300 chars so log stays short
        }})
        # debug for dev
        print(f"\n[LLM INPUT] {prompts[0][:200]}")

    def on_llm_end(self, response, **kwargs):
        logger = get_logger("LLM")
        reply = response.generations[0][0].text
        logger.info("LLM replied", extra={"extra": {
            "reply": reply[:300]
        }})
        # debug for dev
        print(f"[LLM OUTPUT] {reply[:200]}\n")


# 1. IMPORTS
import os
import asyncio                          # <-- needed for asyncio.run()
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field


# 2. ENVIRONMENT SETUP  
load_dotenv(r"C:\Users\Mohamed Arshad\Downloads\My_RAG_Lab\MultiAgent-LLM-Agent-Orchestrator_lab\.env", override=True)
# Only set if the value exists — avoids crash if a key is missing in .env
if os.getenv('GROQ_API_KEY'):      os.environ['GROQ_API_KEY']      = os.getenv('GROQ_API_KEY')
if os.getenv('LANGSMITH_API_KEY'): os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
if os.getenv('OPENAI_API_KEY'):    os.environ['OPENAI_API_KEY']    = os.getenv('OPENAI_API_KEY')
os.environ['LANGSMITH_PROJECT'] = 'Agent_React'


# 3. LLM + CHAINS SETUP  
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    verbose=True,                        
    callbacks=[LoggerCallback()]         
)

#  Question Router 
class RouteQuery(BaseModel):
    datasource: Literal['web_search', 'vectorstore'] = Field(
        description="Given a user question choose to route it to web search or a vectorstore."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)
router_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""),
    ('human', '{question}')
])
question_router = router_prompt | structured_llm_router

# --- Document Grader ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
grader_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing relevance of a retrieved document to a user question.
Give a binary score 'yes' or 'no'."""),
    ('human', 'Retrieved Document:{document}\n\nUser question:{question}\n\n')
])
retrieval_grader = grader_prompt | structured_llm_grader

# --- RAG Generate Chain ---
client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt")
generate_rag_chain = rag_prompt | llm | StrOutputParser()

# --- Hallucination Grader ---
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
hallucination_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing whether an LLM generation is grounded in retrieved facts.
Give a binary score 'yes' or 'no'."""),
    ('human', 'Set of facts:\n\n{documents}\n\nGeneration: {generation}')
])
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

# --- Answer Grader ---
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer addresses/resolves a question. Give 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])
answer_grader = answer_prompt | structured_llm_answer_grader

# --- Question Rewriter ---
question_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer optimizing questions for vectorstore retrieval."),
    ("human", "Here is the initial question: {question}\n\nFormulate an improved question.")
])
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

# --- Web Search Tool ---
web_search_tool = TavilySearch(max_results=5)


# 4. GRAPH STATE
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]


# 5. NODE FUNCTIONS
# Note: retriever is passed in as a parameter
#       because it is built inside async main()

def retrieve(state: GraphState, retriever):
    logger = get_logger("Retrieve")
    question = state['question']
    logger.info("Node started", extra={"extra": {"question": question}})

    documents = retriever.invoke(question)

    logger.info("Retrieval complete", extra={"extra": {
        "question": question,
        "num_docs_retrieved": len(documents),
        "next_node": "grade_documents"
    }})
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    logger = get_logger("Generate")
    question  = state['question']
    documents = state['documents']

    # documents can be a single Document (from web_search) or a list (from retrieve)
    # so we normalise it to always be a list
    if isinstance(documents, Document):
        documents = [documents]

    logger.info("Node started", extra={"extra": {
        "question": question,
        "num_docs_available": len(documents)
    }})

    generation = generate_rag_chain.invoke({"question": question, "context": documents})

    logger.info("Generation complete", extra={"extra": {
        "question": question,
        "generation_preview": generation[:150],
        "next_node": "grade_generation_v_documents_and_question"
    }})
    return {"question": question, "documents": documents, "generation": generation}


def grade_documents(state: GraphState):
    logger = get_logger("GradeDocuments")
    question  = state['question']
    documents = state['documents']
    logger.info("Node started", extra={"extra": {
        "question": question,
        "total_docs": len(documents)
    }})

    filtered_docs = []
    for i, d in enumerate(documents):
        score = retrieval_grader.invoke({"document": d.page_content, "question": question})
        grade = score.binary_score
        logger.debug(f"Document {i+1} graded", extra={"extra": {
            "doc_index": i + 1,
            "relevant": grade,
            "doc_preview": d.page_content[:100]
        }})
        if grade == 'yes':
            filtered_docs.append(d)

    logger.info("Grading complete", extra={"extra": {
        "total_docs": len(documents),
        "relevant_docs_kept": len(filtered_docs),
        "docs_dropped": len(documents) - len(filtered_docs)
    }})
    return {"documents": filtered_docs, "question": question}


def transform_query(state: GraphState):
    logger = get_logger("TransformQuery")
    question  = state['question']
    documents = state['documents']
    logger.info("Node started — rewriting question", extra={"extra": {"original_question": question}})

    better_question = question_rewriter.invoke({"question": question})

    logger.info("Query rewritten", extra={"extra": {
        "original_question": question,
        "rewritten_question": better_question,
        "next_node": "retrieve"
    }})
    return {"documents": documents, "question": better_question}


def web_search(state: GraphState):
    logger = get_logger("WebSearch")
    question = state['question']
    logger.info("Node started", extra={"extra": {"question": question}})

    # docs = web_search_tool.invoke({"query": question})
    # web_results = "\n\n".join([d['content'] for d in docs])
    docs = web_search_tool.invoke(question)
    web_results = "\n\n".join([d['content'] for d in docs['results']])
    web_results = Document(page_content=web_results)

    logger.info("Web search complete", extra={"extra": {
        "question": question,
        "num_results": len(docs['results']),
        "results_preview": [{"title": d['title'], "url": d['url']} for d in docs['results']],
        "next_node": "generate"
        }})
    return {"question": question, "documents": [web_results]}  # always return a list


# 6. CONDITIONAL EDGE FUNCTIONS

def route_question(state: GraphState):
    logger = get_logger("Router")
    question = state['question']
    logger.info("Routing question", extra={"extra": {"question": question}})

    source = question_router.invoke({"question": question})

    logger.info("Routing decision made", extra={"extra": {
        "question": question,
        "routed_to": source.datasource,
    }})

    return "web_search" if source.datasource == "web_search" else "vectorstore"


def decide_to_generate(state: GraphState):
    logger = get_logger("DecideToGenerate")
    filtered_documents = state['documents']

    if not filtered_documents:
        logger.warning("No relevant docs — rewriting query", extra={"extra": {
            "decision": "transform_query",
            "reason": "All retrieved documents were graded as not relevant"
        }})
        return 'transform_query'
    else:
        logger.info("Relevant docs found — generating", extra={"extra": {
            "decision": "generate",
            "num_relevant_docs": len(filtered_documents)
        }})
        return 'generate'


def grade_generation_v_documents_and_question(state: GraphState):
    logger = get_logger("GradeGeneration")
    question   = state['question']
    documents  = state['documents']
    generation = state['generation']

    logger.info("Checking hallucination", extra={"extra": {"question": question}})

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if score.binary_score == "yes":
        logger.info("No hallucination detected")

        score2 = answer_grader.invoke({"question": question, "generation": generation})

        if score2.binary_score == "yes":
            logger.info("Answer is useful — FINAL OUTPUT", extra={"extra": {
                "decision": "useful",
                "generation_preview": generation[:150]
            }})
            return "useful"
        else:
            logger.warning("Answer does not address question", extra={"extra": {"decision": "not useful"}})
            return "not useful"
    else:
        logger.warning("Hallucination detected — retrying", extra={"extra": {"decision": "not supported"}})
        return "not supported"


# 7. ASYNC MAIN — everything with 'await' here
async def main():
    setup_logger = get_logger("Setup")

    # --- Load docs async ---
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    loader = PlaywrightURLLoader(urls=urls)
    docs = await loader.aload()                  # await is safe INSIDE async def
    setup_logger.info("Docs loaded", extra={"extra": {"doc_count": len(docs)}})

    # --- Split + embed ---
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    docs_splits = text_splitter.split_documents(documents=docs)
    setup_logger.info("Docs split", extra={"extra": {"chunk_count": len(docs_splits)}})

    vector_store = FAISS.from_documents(documents=docs_splits, embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    setup_logger.info("Vector store ready")

    # --- Build graph ---
    # retriever is now available so we pass it into retrieve() via lambda
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search",       web_search)
    workflow.add_node("retrieve",         lambda s: retrieve(s, retriever))
    workflow.add_node("grade_documents",  grade_documents)
    workflow.add_node("generate",         generate)
    workflow.add_node("transform_query",  transform_query)

    workflow.add_conditional_edges(START, route_question,
        {'web_search': 'web_search', 'vectorstore': 'retrieve'})

    workflow.add_edge('web_search',      'generate')
    workflow.add_edge('retrieve',        'grade_documents')

    workflow.add_conditional_edges('grade_documents', decide_to_generate,
        {'transform_query': 'transform_query', 'generate': 'generate'})

    workflow.add_edge('transform_query', 'retrieve')

    workflow.add_conditional_edges('generate', grade_generation_v_documents_and_question,
        {'useful': END, 'not useful': 'transform_query', 'not supported': 'generate'})

    app = workflow.compile()

    # --- Run ---
    run_logger = get_logger("Run")

    run_logger.info("=" * 60)
    run_logger.info("Pipeline run 1", extra={"extra": {"question": "What is machine learning"}})
    app.invoke({"question": "What is machine learning"})

    run_logger.info("=" * 60)
    run_logger.info("Pipeline run 2", extra={"extra": {"question": "What is long term memory in agents?"}})
    app.invoke({"question": "What is long term memory and short term memory in agents how they differ and when to use it?"})


# 8. ENTRY POINT
if __name__ == "__main__":
    asyncio.run(main())   # starts the async engine and runs main()
