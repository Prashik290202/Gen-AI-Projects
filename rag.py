import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader 
from PyPDF2 import PdfWriter, PdfReader
from langchain.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
import json
from PyPDF2 import PdfReader
import io
import base64

GPT_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
os.environ["AZURE_OPENAI_API_KEY"] = "ADD YOUR OWN API KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "ADD ENDPOINT"
os.environ["OPENAI_API_VERSION"] = "2023-02-03"
 

embeddings = AzureOpenAIEmbeddings(
    model="Embedding-ada",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
) 
# File to store chat history
CHAT_HISTORY_FILE = "chat_history.json"
TEMP_DIR = os.path.join(os.getcwd(), "temp_uploads")  

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def read_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)  
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read()) 

        
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())  
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(file_path)  
            docs.extend(loader.load())  

    return docs


# Function to split text into chunks for processing
def get_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    return chunks

# Create a local vector store
def vector_store(text_chunks):
    
    if os.path.exists("faiss_db"):
        vector_store = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(text_chunks)
        
    else:
        
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    
    
    vector_store.save_local("faiss_db")

# Template for the bot's responses
prompt_template = """You are a helpful assistant. Answer the question as detailed as possible from the provided context.
    If the answer is not available in the provided context, say "answer is not available in the context" and do not provide incorrect answers.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}"""

# Function to handle conversational chain
def get_conversational_chain(retriever, ques, chat_history):
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version="2024-02-15-preview",
        azure_deployment=GPT_DEPLOYMENT_NAME,
    )

    # Initialize memory with chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    if chat_history:
        for user_question, answer in chat_history:
            memory.chat_memory.add_user_message(user_question)
            memory.chat_memory.add_ai_message(answer)

    # Define the chat prompts using templates
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages)

    # ConversationalRetrievalChain for handling responses
    bot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

    result = bot.invoke({"question": ques, "chat_history": chat_history})

    # Prepare citation references
    citation_ref = []
    for chunk_no, chunk_doc in enumerate(result.get('source_documents', [])):
        response_citation = {
            'chunk_number': chunk_no+1,
            'pagecontent': chunk_doc.page_content,
            'page_ref_number': (chunk_doc.metadata['page'] + 1) if 'page' in chunk_doc.metadata else 'none',   
            'source_filename': os.path.basename(chunk_doc.metadata['source'])
        }
        citation_ref.append(response_citation)

    
    memory.chat_memory.add_user_message(ques)
    memory.chat_memory.add_ai_message(result['answer'])

    return result, citation_ref  

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f)

def displayPDF(file, page_number):
    with open(file, "rb") as f:
        reader = PdfReader(f)
        if page_number < 1 or page_number > len(reader.pages):
            return None  
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        temp_pdf = io.BytesIO()
        writer.write(temp_pdf)
        temp_pdf.seek(0)  

        base64_pdf = base64.b64encode(temp_pdf.read()).decode('utf-8')
    
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page_number}" width="700" height="1000" type="application/pdf">'
    return pdf_display


def user_input(user_question, chat_history):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity_score_threshold", 
                                 search_kwargs={"score_threshold":0.65,"k":4
                                                
                                                })

    result, citation_ref = get_conversational_chain(retriever, user_question, chat_history)

    if result:
        st.write("Response: ", result['answer'])
        with st.expander("Click to view Citation", expanded=False):
            if citation_ref:
                st.subheader("Citation References")
                for ref in citation_ref:
                    st.write(f"**Chunk Number:** {ref['chunk_number']}")
                    st.write(f"**Page Content:** {ref['pagecontent']}")
                    st.write(f"**Page Reference Number:** {ref['page_ref_number']}")
                    st.write(f"**Source Filename:** {ref['source_filename']}")

                    file_path = os.path.join(TEMP_DIR, ref['source_filename'])
                    file_extension = os.path.splitext(ref['source_filename'])[1].lower()
                    if os.path.exists(file_path):
                        if file_extension == '.pdf':
                            page_number = ref['page_ref_number']  
                            pdf_link = displayPDF(file_path, page_number)
                            if pdf_link:  
                                st.markdown(pdf_link, unsafe_allow_html=True)
                            else:
                                st.write("Invalid page number or PDF not found.")
                        
                    else:
                        st.write("File not found.")
                    
                    st.write("---")

        
        chat_history.append((user_question, result['answer']))

    return chat_history


def main():
    st.set_page_config("Chat PDF /TEXT")
    st.header("RAG ChatBot with PDF")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    user_question = st.text_input("Ask a Question....")

    if st.button("Submit"):
        if user_question:
            st.session_state.chat_history = user_input(user_question, st.session_state.chat_history)
            save_chat_history(st.session_state.chat_history)  

    with st.expander("Click to view Chat History", expanded=False):
        st.subheader("Chat History")
        for user_question, ai_response in st.session_state.chat_history:
            st.write(f"User: {user_question}")
            st.write(f"AI: {ai_response}")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF and Text Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                docs = read_files(uploaded_files)  
                text_chunks = get_chunks(docs)  
                vector_store(text_chunks)  
                st.success("Done")

     
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []  
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)  
            st.success("Chat history cleared!")  

if __name__ == "__main__":
    main()