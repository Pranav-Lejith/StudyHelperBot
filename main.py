import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import requests
from requests.exceptions import HTTPError

# Define subjects dictionary
subjects = {
    "DemocraticPolicies": {
        "Chapter 1 - What is democracy? Why democracy?": "chapter1.pdf",
        "Chapter 2 - Constitutional Design": "chapter2.pdf",
        "Chapter 3 - Electoral Politics": "chapter3.pdf",
        "Chapter 4 - Working of Institutions": "chapter4.pdf",
        "Chapter 5 - Democratic Rights": "chapter5.pdf"
    },
    "Economics": {
        "Chapter 1 - The Story of Village Palampur": "chapter1.pdf",
        "Chapter 2 - People as Resource": "chapter2.pdf",
        "Chapter 3 - Poverty as a Challenge": "chapter3.pdf",
        "Chapter 4 - Food Security In India": "chapter4.pdf"
    },
    "English_beehive": {
        "Chapter 1 - The Fun They Had": "chapter1.pdf",
        "Chapter 2 - The Sound of Music": "chapter2.pdf",
        "Chapter 3 - The Little Girl": "chapter3.pdf",
        "Chapter 4 - A Truly Beautiful Mind": "chapter4.pdf",
        "Chapter 5 - The Snake and the Mirror": "chapter5.pdf",
        "Chapter 6 - My Childhood": "chapter6.pdf",
        "Chapter 7 - Packing": "chapter7.pdf",
        "Chapter 8 - Kathmandu": "chapter8.pdf",
        "Chapter 9 - If I were you": "chapter9.pdf"
    },
    "English_moments": {
        "Chapter 1 - The Lost Child": "chapter1.pdf",
        "Chapter 2 - The Adventures of Toto": "chapter2.pdf",
        "Chapter 3 - Iswaran the Storyteller": "chapter3.pdf",
        "Chapter 4 - In the Kingdom of Fools": "chapter4.pdf",
        "Chapter 5 - The Happy Prince": "chapter5.pdf",
        "Chapter 6 - Weathering the storm in Erasma": "chapter6.pdf",
        "Chapter 7 - The Last Leaf": "chapter7.pdf",
        "Chapter 8 - A house is not a Home": "chapter8.pdf",
        "Chapter 9 - The Beggar": "chapter9.pdf"
    },
    "Geography": {
        "Chapter 1 - India - Size and Location": "chapter1.pdf",
        "Chapter 2 - India - Physical Features": "chapter2.pdf",
        "Chapter 3 - Drainage": "chapter3.pdf",
        "Chapter 4 - Climate": "chapter4.pdf",
        "Chapter 5 - Natural Vegetation and Wildlife": "chapter5.pdf",
        "Chapter 6 - Population": "chapter6.pdf"
    },
    "History": {
        "Chapter 1 - The French Revolution": "chapter1.pdf",
        "Chapter 2 - Russian Revolution": "chapter2.pdf",
        "Chapter 3 - Nazism and Rise of Hitler": "chapter3.pdf",
        "Chapter 4 - Forest Society and Colonialism": "chapter4.pdf",
        "Chapter 5 - Pastoralists in the Modern World": "chapter5.pdf"
    },
    "Maths": {
        "Chapter 1 - Number Systems": "chapter1.pdf",
        "Chapter 2 - Polynomials": "chapter2.pdf",
        "Chapter 3 - Coordinate Geometry": "chapter3.pdf",
        "Chapter 4 - Linear equations in two variables": "chapter4.pdf",
        "Chapter 5 - Euclid's Geometry": "chapter5.pdf",
        "Chapter 6 - Lines and Angles": "chapter6.pdf",
        "Chapter 7 - Triangles": "chapter7.pdf",
        "Chapter 8 - Quadrilaterals": "chapter8.pdf",
        "Chapter 9 - Circles": "chapter9.pdf",
        "Chapter 10 - Heron's Formula": "chapter10.pdf",
        "Chapter 11 - Surface Areas and Volumes": "chapter11.pdf",
        "Chapter 12 - Statistics": "chapter12.pdf"
    },
    "Science": {
        "Chapter 1 - Matter in Our Surroundings": "chapter1.pdf",
        "Chapter 2 - Is Matter Around Us Pure": "chapter2.pdf",
        "Chapter 3 - Atoms and Molecules": "chapter3.pdf",
        "Chapter 4 - Structure of the Atom": "chapter4.pdf",
        "Chapter 5 - The Fundamental Unit of Life": "chapter5.pdf",
        "Chapter 6 - Tissues": "chapter6.pdf",
        "Chapter 7 - Motion": "chapter7.pdf",
        "Chapter 8 - Force and Laws of Motion": "chapter8.pdf",
        "Chapter 9 - Gravitation": "chapter9.pdf",
        "Chapter 10 - Work and Energy": "chapter10.pdf",
        "Chapter 11 - Sound": "chapter11.pdf",
        "Chapter 12 - Improvement in Food Resources": "chapter12.pdf"
    }
}

# Fetch API keys from Streamlit secrets
api_keys = [
    st.secrets["api_keys"]["google_key_1"],
    st.secrets["api_keys"]["google_key_2"],
    st.secrets["api_keys"]["google_key_3"]
]

# Initialize session state attributes if they don't exist
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'font_size' not in st.session_state:
    st.session_state.font_size = 16

def get_current_api_key():
    return api_keys[st.session_state.current_api_key_index]

def rotate_api_key():
    st.session_state.current_api_key_index = (st.session_state.current_api_key_index + 1) % len(api_keys)

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(pdf_path):
    text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_current_api_key())
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an expert in various subjects including History, Geography, English, Economics, Maths, and Science. Answer the 
    question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not 
    in the provided context, just say, "The answer is not available in the context," and do not provide incorrect information.

    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=get_current_api_key())
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, pdf_path=None):
    if pdf_path:
        text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_current_api_key())
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_current_api_key())
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except HTTPError as http_err:
        if http_err.response.status_code == 429:
            rotate_api_key()
            return user_input(user_question, pdf_path)
        else:
            return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"Other error occurred: {err}"

def display_chat():
    for message in st.session_state.messages:
        role = "User" if message['role'] == 'user' else "Assistant"
        st.markdown(f"**{role}:** {message['content']}")

def main():
    st.set_page_config(page_title="PDF Reader", page_icon=":book:", layout="wide")

    st.sidebar.header("📚 PDF Reader")
    st.sidebar.markdown("""
    ### Choose an Option
    """)

    option = st.sidebar.selectbox("Select Option", ["Select Subject and Chapter", "General Question"], index=0)

    if option == "Select Subject and Chapter":
        subject = st.sidebar.selectbox("Select Subject", list(subjects.keys()))
        chapter = st.sidebar.selectbox("Select Chapter", list(subjects[subject].keys()))
        pdf_path = os.path.join(subject, subjects[subject][chapter])
    else:
        pdf_path = None

    st.sidebar.markdown("---")
    
    st.sidebar.markdown("""
    ### How It Works 🛠️

    1. **Choose Your Option:**  
       On the sidebar, you can select either "Select Subject and Chapter" or "General Question."

    2. **Select Subject and Chapter:**  
       If you choose this option, select the subject and chapter from the dropdowns. This will load the relevant PDF file.

    3. **Enter Your Question:**  
       Type your question into the input box and click "Submit."

    4. **Processing Your Question:**  
       The system will process your question using the relevant PDF context or the general context.

    5. **Receiving the Answer:**  
       The answer is displayed in the chat interface along with your question for reference.

    6. **Interaction History:**  
       All interactions, including your questions and the system's responses, are displayed in the chat interface for easy reference.
    """)

    st.sidebar.info("**Created by Pranav Lejith (Amphibiar)**")
    st.sidebar.info("**This bot answers questions from all the 9th standard textbooks except L2 and Artificial Intelligence**")

    font_size = st.sidebar.slider("Font Size", min_value=10, max_value=30, value=16)

    st.session_state.font_size = font_size

    st.title("PDF Reader(Answers questions from 9th Standard Textbooks)")

    user_question = st.text_input("Enter your question:")

    if st.button("Submit"):
        if pdf_path:
            answer = user_input(user_question, pdf_path)
        else:
            answer = user_input(user_question)

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown(f"<style>div[data-testid='stText'] {{ font-size: {st.session_state.font_size}px; }}</style>", unsafe_allow_html=True)
    
    display_chat()

if __name__ == "__main__":
    main()
