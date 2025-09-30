import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
import asyncio

# --- PROJECT SETUP ---
# For a professional resume project, it's best to use Streamlit's secrets management.
# Create a file .streamlit/secrets.toml and add your keys there:
# OPENAI_API_KEY = "sk-..."
# TAVILY_API_KEY = "tvly-..."
#
# If you are running locally without Streamlit Cloud, you can use a .env file.
# Create a .env file in your root directory and add:
# OPENAI_API_KEY="sk-..."
# TAVILY_API_KEY="tvly-..."
#
# The code below will try to load from .env if secrets are not available.
try:
    # Try to get from Streamlit secrets first
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except (KeyError, AttributeError):
    # Fallback to loading from .env file for local development
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
        st.error("API keys for OpenAI and Tavily are not set. Please add them to your .env file or Streamlit secrets.")
        st.stop()

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="CogniGraph: Self-Correcting Insight Engine",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CogniGraph: Self-Correcting Insight Engine")
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        .stButton > button {
            border-radius: 20px;
            border: 1px solid #4B8BBE;
            background-color: #4B8BBE;
            color: white;
        }
        .stButton > button:hover {
            border: 1px solid #3A6A94;
            background-color: #3A6A94;
            color: white;
        }
        .stChatMessage {
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("About CogniGraph")
st.sidebar.info(
    "This is a self-correcting Retrieval-Augmented Generation (RAG) system. "
    "It uses a graph-based approach to answer questions. If its initial document retrieval is not relevant, "
    "it corrects itself by rewriting the query and searching the web."
)
st.sidebar.markdown("**Technology Stack:**")
st.sidebar.markdown("- Streamlit\n- LangChain & LangGraph\n- OpenAI (GPT-3.5)\n- FAISS Vector Store\n- Tavily Search API")

# --- CACHED RESOURCES ---
@st.cache_resource
def build_retriever():
    """Builds and caches the document retriever."""
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

retriever = build_retriever()

# --- RAG MODELS AND TOOLS (CACHED) ---
@st.cache_resource
def get_tools_and_chains():
    """Initializes and caches all necessary models, tools, and chains."""
    # Retrieval Grader
    from pydantic import BaseModel, Field
    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # RAG Chain
    prompt = hub.pull("rlm/rag-prompt")
    rag_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = prompt | rag_llm | StrOutputParser()

    # Question Rewriter
    system_rewriter = """You a question re-writer that converts an input question to a better version that is optimized
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [("system", system_rewriter), ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Web Search Tool
    web_search_tool = TavilySearchResults(k=3)
    
    return retrieval_grader, rag_chain, question_rewriter, web_search_tool

retrieval_grader, rag_chain, question_rewriter, web_search_tool = get_tools_and_chains()


# --- LANGGRAPH SETUP ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# --- NODES ---
def retrieve(state):
    st.session_state.status_log.append("🔎 Retrieving relevant documents...")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    st.session_state.status_log.append("🧠 Generating final answer...")
    question = state["question"]
    documents = state["documents"]
    generation_stream = rag_chain.stream({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation_stream}

def grade_documents(state):
    st.session_state.status_log.append("✅ Grading document relevance...")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            st.session_state.status_log.append(f"    - Document relevant: Found content on '{d.metadata.get('title', '...')[:30]}...'")
            filtered_docs.append(d)
        else:
            st.session_state.status_log.append(f"    - Document NOT relevant. Flagging for web search.")
            web_search = "Yes"
    
    # If all documents are irrelevant, we will perform a web search
    if not filtered_docs:
         st.session_state.status_log.append("⚠️ No relevant documents found in local index.")

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    st.session_state.status_log.append("✍️ Rewriting question for web search...")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    st.session_state.status_log.append(f"    - New question: {better_question}")
    return {"documents": documents, "question": better_question}

def web_search(state):
    st.session_state.status_log.append("🌐 Performing web search...")
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

# --- EDGES ---
def decide_to_generate(state):
    st.session_state.status_log.append("🤔 Assessing graded documents...")
    web_search = state["web_search"]
    if web_search == "Yes":
        st.session_state.status_log.append("    - Decision: All documents were not relevant. Transforming query for web search.")
        return "transform_query"
    else:
        st.session_state.status_log.append("    - Decision: Relevant documents found. Proceeding to generate answer.")
        return "generate"

@st.cache_resource
def build_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"}
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

app = build_graph()

# --- STREAMLIT CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status_log" not in st.session_state:
    st.session_state.status_log = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for doc in message["sources"]:
                        st.info(f"Source: {doc.metadata.get('source', 'Web Search')}\n\nContent: {doc.page_content[:350]}...")
        else:
            st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything about AI agents, prompt engineering, or LLM attacks..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.status_log = []

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        
        # This function updates the status display
        async def update_status():
            while True:
                if st.session_state.get('graph_finished', False):
                    break
                status_text = "Thinking...\n\n" + "\n".join(st.session_state.status_log)
                status_placeholder.status(status_text, expanded=True)
                await asyncio.sleep(0.1)

        st.session_state['graph_finished'] = False
        update_task = asyncio.create_task(update_status())

        # Run the graph
        inputs = {"question": prompt}
        final_state = None
        stream = app.stream(inputs)
        
        for output in stream:
            for key, value in output.items():
                if key == "generate":
                    final_state = value
        
        # Stop the status update and show the final answer
        st.session_state['graph_finished'] = True
        update_task.cancel()
        status_placeholder.empty()

        if final_state and "generation" in final_state:
            # Stream the final answer
            full_response = answer_placeholder.write_stream(final_state["generation"])
            
            # Add full response and sources to session state
            assistant_message = {"role": "assistant", "content": full_response, "sources": final_state.get("documents", [])}
            st.session_state.messages.append(assistant_message)
            
            # Display sources in an expander after the answer is complete
            with st.expander("View Sources"):
                if final_state.get("documents"):
                    for doc in final_state["documents"]:
                        st.info(f"**Source:** {doc.metadata.get('source', 'Web Search')}\n\n**Content:** {doc.page_content[:350]}...")
                else:
                    st.warning("No sources were used for this answer.")
        else:
            error_message = "Sorry, I encountered an error and couldn't generate a response."
            answer_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
