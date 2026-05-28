import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEmbeddings , HuggingFaceEndpoint
from langgraph.graph import StateGraph , START , END
from typing import Optional , TypedDict , Literal , Annotated
import operator
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage , BaseMessage
from langchain_groq import ChatGroq
import uuid
load_dotenv(".env")

# Streamlit Cloud: push st.secrets into env vars
if hasattr(st, "secrets"):
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))

st.title("Doc Explainer")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf" , "docx"],
    accept_multiple_files=False
)

if uploaded_file:
    st.write("File uploaded ✅")

    os.makedirs("docs", exist_ok=True)
    save_path = f"docs/{uploaded_file.name}"

    #Saving the Document
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "message" not in st.session_state:
    st.session_state.message = []

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


for message in st.session_state.message:
    with st.chat_message(message["role"]):
        st.write(message["content"])



@st.cache_resource
def get_model():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API"),
        model_name="llama-3.1-8b-instant"
    )

    return llm


@st.cache_resource
def embedding():

    #Embedding Model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    return embedding

model = get_model()
embedding = embedding()

if uploaded_file:

    if st.session_state.vectorstore is None:

        loader = PyPDFLoader(save_path)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=350
        )

        docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            docs,
            embedding=embedding
        )

        st.session_state.vectorstore = vectorstore

    else:
        vectorstore = st.session_state.vectorstore

question = st.chat_input("Ask Question About Document")

#State
class DocState(TypedDict):

    question : str
    response : str
    content : str
    history : Annotated[list[BaseMessage], operator.add]

memory = MemorySaver()

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

def retrived(state: DocState):

    question = state['question']

    retriver = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriver.invoke(question)

    content = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        'content': content
    }

def LLM_mod(state: DocState):

    question = state['question']
    history = state['history']
    content = state['content']
    formatted_history = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted_history += f"User: {msg.content}\n"
        else:
            formatted_history += f"Assistant: {msg.content}\n"
    prompt = f"""
    You are a helpful document assistant.

    Use the provided document content and chat history
    to answer the user question.

    Chat History:
    {formatted_history}

    Document Content:
    {content}

    Question:
    {question}
    """

    result = model.invoke(prompt)
    human_msg = HumanMessage(content=question)
    response = result.content

    return {
        'response': response,
        'history': [human_msg , result]
    }

#Graph
graph = StateGraph(DocState)

#Nodes
graph.add_node('retriver' , retrived)
graph.add_node('llm' , LLM_mod)

#Edges
graph.add_edge(START , 'retriver')
graph.add_edge('retriver' , 'llm')
graph.add_edge('llm' , END)

graph = graph.compile(checkpointer=memory)

if question and st.session_state.vectorstore:

    #User writting
    st.chat_message('user').write(question)

    st.session_state.message.append({
        "role": "user",
        "content": question
    })

    #AI Response
    state = graph.invoke(
        {'question': question , 'history': st.session_state.history} ,
        config=config
    )
    st.session_state.history = state['history'] 
    st.chat_message('assistant').write(state['response'])

    st.session_state.message.append({
        "role": "assistant",
        "content": state['response']
    })