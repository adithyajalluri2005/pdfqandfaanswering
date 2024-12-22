import os
import uuid
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb.config import Settings

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Chat with your pdf")
st.write("Upload your PDF")

uploaded_pdf = st.file_uploader("Choose a PDF File", accept_multiple_files=False)

with st.sidebar:
    if 'sessions' not in st.session_state:
        st.session_state.sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None

    def create_new_session():
        new_session_id = str(uuid.uuid4())
        st.session_state.sessions[new_session_id] = ChatMessageHistory()
        st.session_state.current_session = new_session_id

    def delete_session(session_id):
        del st.session_state.sessions[session_id]
        if st.session_state.current_session == session_id:
            st.session_state.current_session = None

    if st.button("New Session"):
        create_new_session()

    sessions_list = list(st.session_state.sessions.keys())
    if sessions_list:
        st.selectbox("Select Session", sessions_list, key="session_select")
        st.session_state.current_session = st.session_state.session_select
        if st.button("Delete Session"):
            delete_session(st.session_state.current_session)
    else:
        st.write("No sessions available")

if uploaded_pdf and st.session_state.current_session:
    session_id = st.session_state.current_session
    temppdf = "./temp.pdf"
    with open(temppdf, "wb") as file:
        file.write(uploaded_pdf.getvalue())

    loader = PyPDFLoader(temppdf)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)


    vectorstore = Chroma(persist_directory="./chroma_db1", embedding_function=embeddings)
    retriever = vectorstore.as_retriever() 
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


    system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\{context}"""



    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        return st.session_state.sessions[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )



    user_input = st.text_input("Enter Your Query")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.write("Assistant:", response['answer'])
else:
    st.write("Upload a PDF")