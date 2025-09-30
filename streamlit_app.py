import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# 🔐 Hugging Face Token (store this in Streamlit secrets when deploying)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 📄 Load and split your data
loader = TextLoader("hmpidata.txt", encoding="utf-8")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 🔍 Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 🧠 LLM via Hugging Face Hub (Gemma 2B or any hosted model)
llm = HuggingFaceHub(repo_id="google/gemma-2b", model_kwargs={"temperature": 0.7, "max_new_tokens": 512})

# 🔁 Memory + Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)

# 🎨 Streamlit UI
st.title("🧠 HMPI Chatbot (Gemma-powered)")
user_input = st.text_input("You:", key="input")

if user_input:
    response = qa_chain.run(user_input)
    st.markdown(f"**Bot:** {response}")