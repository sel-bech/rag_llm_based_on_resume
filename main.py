import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# custom prompt template
template = """
You are a helpful AI assistant answering questions about a candidate based on their resume.
Use the following pieces of context (the resume) to answer the question at the end.
If you don't know the answer, just say that the information is not present in the resume.
Be professional and concise.

Context: 
{context}

Question: {question}

Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def main():
    load_dotenv()
    print("Loading Resume PDF...")
    # Load the PDF
    try:
        # Make sure you have your resume file named 'mycv.pdf' in the same directory
        loader = PyPDFLoader("mycv.pdf")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    print(f"Loaded {len(documents)} pages.")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks.")

    if not all_splits:
        print("No text found in PDF.")
        return

    print("Creating embeddings and vector store...")
    # Create embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Create vector store
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    except Exception as e:
        print(f"Error creating embeddings/vector store: {e}")
        return

    # Create the LLM
    # Using Groq's API with Llama3-70b-8192 model (you can change this to mixtral-8x7b-32768, etc.)
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in .env file.")
        return

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile"
    )

    # Create the RetrievalQA chain
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    print("\n--- Model Ready! Ask questions about the candidate's resume ---")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # Determine language (simple heuristic) and adjust prompt slightly if needed, 
        # but the model usually handles it well. 
        # We can add a system prompt via a custom prompt template if strict adherence to Arabic is needed.
        
        result = rag_chain.invoke(query)
        
        print(f"\nAnswer: {result}\n")
        
        # Optional: Print sources
        # print("Sources:")
        # for doc in result['source_documents']:
        #     print(f"- Page {doc.metadata['page']}: {doc.page_content[:100]}...")
        # print("\n")

if __name__ == "__main__":
    main()
