import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorStore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_llm_client():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in .env file!")
        return None
    
    client = InferenceClient(token=token)
    return client

def get_answer(question, context, chat_history, client):
    if client is None:
        return "Error: LLM client not initialized. Please check your API token."
    
    history_text = ""
    if chat_history:
        for q, a in chat_history[-2:]:
            history_text += f"Q: {q}\nA: {a}\n\n"
    
    system_message = f"""You are a helpful assistant that answers questions based on the provided document context. 
Be concise and specific. If the answer is not in the context, say so.

Context from documents:
{context[:3000]}"""

    user_message = f"""{history_text}Question: {question}

Please provide a clear answer based on the context above."""
    
    models_to_try = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3-mini-4k-instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    for model in models_to_try:
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=512,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            return answer.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "not supported" in error_msg.lower():
                try:
                    full_prompt = f"{system_message}\n\n{user_message}"
                    response = client.summarization(full_prompt[:1024], model="facebook/bart-large-cnn")
                    return response.summary_text
                except:
                    pass
            
            if model == models_to_try[-1]:
                return f"Unable to generate response. Error: {error_msg}"
            continue
    
    return "Unable to generate response. Please try again."

def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process PDF documents first!")
        return
    
    if st.session_state.llm_client is None:
        st.error("LLM client not initialized. Please check your API token.")
        return
    
    with st.spinner("Thinking..."):
        try:
            docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return
        
        answer = get_answer(
            user_question, 
            context, 
            st.session_state.chat_history,
            st.session_state.llm_client
        )
        
        st.session_state.chat_history.append((user_question, answer))
    
    st.session_state.display_count = len(st.session_state.chat_history)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "display_count" not in st.session_state:
        st.session_state.display_count = 0
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
    
    for question, ans in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", ans), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
            st.success("API Token loaded")
        else:
            st.error("API Token missing! Add HUGGINGFACEHUB_API_TOKEN to your .env file")
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)

                        st.session_state.vectorstore = get_vectorStore(text_chunks)
                        
                        st.session_state.llm_client = get_llm_client()
                        
                        st.session_state.chat_history = []
                        
                        st.success(f"Processed {len(text_chunks)} text chunks. You can now ask questions!")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == '__main__':
    main()