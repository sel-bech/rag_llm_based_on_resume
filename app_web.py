import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PDF_FILE = "mycv.pdf"

# Initialize env
load_dotenv()

# Custom prompt template
template = """
You are the candidate described in the resume. Answer questions as if you are this person, using first-person pronouns ("I", "me", "my").
Use the following pieces of context (your resume) to answer the question at the end.
If the answer is not in the context, politely state that you haven't included that detail in your resume.
Be professional, confident, and concise.

Context: 
{context}

Question: {question}

Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

st.set_page_config(page_title="My Resume AI", page_icon="👨‍💼")

st.title("👨‍💼 Chat with Me (AI Version)")
st.markdown("I am an AI representation of the candidate. Ask me anything about my experience, skills, and background!")

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initializes the RAG pipeline by loading the PDF, processing it, 
    creating embeddings, and setting up the vector store.
    This function is cached to avoid reloading everything on every interaction.
    """
    if not os.path.exists(PDF_FILE):
        st.error(f"File not found: {PDF_FILE}")
        return None

    with st.spinner("Loading PDF and preparing AI model..."):
        # Load the PDF
        try:
            loader = PyPDFLoader(PDF_FILE)
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return None

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(documents)

        if not all_splits:
            st.error("No text found in PDF.")
            return None

        # Create embeddings
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            # Create vector store
            vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        except Exception as e:
            st.error(f"Error creating embeddings/vector store: {e}")
            return None
        
        return vectorstore

def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in .env file.")
        return None
    
    return ChatGroq(
        temperature=0,
        model_name=MODEL_NAME
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG Pipeline
vectorstore = initialize_rag_pipeline()

if vectorstore:
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    
    if llm:
        # Create RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | QA_CHAIN_PROMPT
            | llm
            | StrOutputParser()
        )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask me a question about my experience..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = rag_chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
