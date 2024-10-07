import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Set page title and favicon (Ananda logo)
logo_path = "https://ananda.exchange/wp-content/uploads/2022/03/cropped-Fondos-y-recursos-20.png"  # Replace with the path to your Ananda logo
st.set_page_config(
    page_title="ChatBot",
    page_icon=logo_path  # Setting the favicon (logo) in the tab
)

# Access the secrets via st.secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_project = st.secrets["LANGCHAIN_PROJECT"]

# Set environment variables for Langchain
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = langchain_project
os.environ["LANGCHAIN_TRACING_V2"]="true"

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and also Imagine yourself an an expert in the field of cryptocurrency.
    Please provide the most accurate response based on the question.
    If you don't know the answer, try to give generic answer.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("Docs")  # Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Inject CSS with chat UI styles and background image
st.markdown(
    f"""
    <style>
   body {{
        background-image: url("https://ananda.exchange/wp-content/uploads/2023/02/We-Make-Buying-Bitcoin.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}

    .stApp {{
        background-image: url("https://ananda.exchange/wp-content/uploads/2023/02/We-Make-Buying-Bitcoin.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .chat-container {{
        max-width: 700px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        background-color: rgba(0, 0, 0, 0.7); /* Add a semi-transparent background */
    }}

    .user-msg {{
        background-color: #A9A9A9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 45%;
        margin-left: auto;
        text-align: right;
        clear: both;
        word-wrap: break-word;
    }}

    .bot-msg {{
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 45%;
        margin-right: auto;
        text-align: left;
        clear: both;
        word-wrap: break-word;
    }}

    .scrollable-chat {{
        max-height: 400px;
        overflow-y: auto;
        padding-right: 15px;
    }}

    .input-area {{
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #111;
        padding: 10px;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.2);
    }}

    .input-box {{
        width: 80%;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #ccc;
        background-color: black;
        color: white;
    }}

    .send-button {{
        position: relative;
        right: 10px;
        top: -50%;
        background-color: #007bff;
        border: none;
        border-radius: 50%;
        padding: 10px;
        cursor: pointer;
    }}

    .send-button img {{
        width: 20px;
        height: 20px;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Display the logo at the top left corner using st.image
logo_path = "https://ananda.exchange/wp-content/uploads/2022/03/cropped-Fondos-y-recursos-20.png"
st.image(logo_path, width=100)  # Adjust the width if needed

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to introduce the chatbot
def chatbot_greeting():
    st.session_state.chat_history.append({
        "user": "",
        "bot": "I'm Ananda! Here to teach you all about CryptoWorld. Ask me anything!"
    })

# Main UI components
st.title("Master Crypto with ANANDA")

# Node Activation Button (Conversation opener)
if st.button("Click here to Start!"):
    create_vector_embedding()
    chatbot_greeting()  # Add chatbot greeting message
    st.write("Conversation Started!")

# Chat Container to display the conversation history
chat_placeholder = st.empty()  # Placeholder for chat history

# Function to render chat history dynamically
def render_chat():
    chat_html = '<div class="chat-container scrollable-chat">'
    for chat in st.session_state.chat_history:
        if chat['user']:
            chat_html += f'<div class="user-msg">You: {chat["user"]}</div>'
        if chat['bot']:
            chat_html += f'<div class="bot-msg">Ananda: {chat["bot"]}</div>'
    chat_html += '</div>'
    chat_placeholder.markdown(chat_html, unsafe_allow_html=True)

# Render the chat initially
render_chat()

# Fixed input area at the bottom
with st.container():
    user_prompt = st.text_input("Message Ananda", key="message_input", placeholder="Type your message here...", label_visibility="collapsed")

# Process user input
if user_prompt:
    start = time.process_time()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': user_prompt})
    response_text = response['answer']

    # Append the new user message and the response to chat history
    st.session_state.chat_history.append({"user": user_prompt, "bot": response_text})

    # Render the updated chat history
    render_chat()

    # Debugging the response time
    print(f"Response time: {time.process_time() - start}")

# Document similarity section
if user_prompt:
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
