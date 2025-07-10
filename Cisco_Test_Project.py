import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # <-- Use Groq LLM

@st.cache_resource
def setup_qa():
    # 1. Scrape webpage
    url = "https://www.cisco.com/c/en/us/products/collateral/software/ea-faq.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    web_texts = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]

    # 2. Extract PDF text
    pdf_dir = "Cisco_PDFs"
    pdf_texts = []
    if os.path.exists(pdf_dir):
        for fname in os.listdir(pdf_dir):
            if fname.lower().endswith('.pdf'):
                try:
                    loader = PyPDFLoader(os.path.join(pdf_dir, fname))
                    pdf_texts.extend([doc.page_content for doc in loader.load()])
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

    # 3. Combine and split texts
    all_texts = web_texts + pdf_texts
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(all_texts)

    # 4. Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # 5. Set up Groq LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-actual-groq-api-key")  # <-- Add this line
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"  # Or try "mixtral-8x7b-32768"
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa

st.title("Cisco RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa = setup_qa()

user_input = st.text_input("You:", key="input")
if st.button("Send") and user_input:
    answer = qa({"query": user_input})["result"]
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
