import streamlit as st
from dotenv import load_dotenv  # for loading environment variables
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Existing imports
from langchain_community.embeddings import OllamaEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load environment variables
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Load cross-encoder model and tokenizer
cross_encoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # or another suitable model
tokenizer = BertTokenizer.from_pretrained(cross_encoder_model_name)
model = BertForSequenceClassification.from_pretrained(cross_encoder_model_name)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def get_vectorstore(text_chunks):
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=OllamaEmbeddings(model='nomic-embed-text', temperature=0.5, show_progress=True),
        collection_name='local-rag'
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    local_model = 'mistral'
    llm = ChatOllama(model=local_model)
   
    QUERY_PROMPT = PromptTemplate(          
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. 
        Original question: {question}
        Only provide the query, do not do numbering at the start of the questions.
        """
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vectorstore.as_retriever(search_kwargs={"k":5}),
        prompt=QUERY_PROMPT,
        llm=llm
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def rerank_documents(query, documents):
    scores = []
    for doc in documents:
        inputs = tokenizer.encode_plus(query, doc.page_content, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        score = outputs.logits.squeeze().item()
        scores.append((score, doc))
    
    # Sort documents by score
    sorted_docs = [doc for _, doc in sorted(scores, reverse=True)]
    return sorted_docs

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    # Extract and re-rank documents
    documents = [doc for doc in st.session_state.conversation.retriever.get_relevant_documents(user_question)]
    ranked_documents = rerank_documents(user_question, documents)

    # Display responses
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with your DATA (PDFs) using RAG",
                       page_icon=":parrot:")
    st.write(css, unsafe_allow_html=True)
    st.title("💬 Chatbot powered by Ollama")
    st.caption("🚀 Designed by :red[ALSAFAK KAMAL @ IIT_MADRAS]")
    st.caption("🚫 Make sure that you have already downloaded the OLLAMA software in your system")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your DATA (PDFs) using RAG :parrot:")
    user_question = st.text_input("Ask anything related to your Data:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Kindly upload your PDFs below & click on 'Upload'", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Take a deep breath for a While..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
