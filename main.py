import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Load the API key from Streamlit secrets
api_key = st.secrets["google"]["api_key"]
if not api_key:
    raise ValueError("API key not found in Streamlit secrets.")

# Define the subjects and their chapters with readable names
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
        "Chapter 8 - The Little House": "chapter8.pdf",
        "Chapter 9 - A Truly Beautiful Mind": "chapter9.pdf"
    },
    "English_moments": {
        "Chapter 1 - The Lost Child": "chapter1.pdf",
        "Chapter 2 - The Adventures of Toto": "chapter2.pdf",
        "Chapter 3 - Iswaran the Storyteller": "chapter3.pdf",
        "Chapter 4 - In the Kingdom of Fools": "chapter4.pdf",
        "Chapter 5 - The Happy Prince": "chapter5.pdf",
        "Chapter 6 - A House is Not a Home": "chapter6.pdf",
        "Chapter 7 - The Beggar": "chapter7.pdf",
        "Chapter 8 - The Road Not Taken": "chapter8.pdf",
        "Chapter 9 - The Last Leaf": "chapter9.pdf"
    },
    "Geography": {
        "Chapter 1 - Resources and Development": "chapter1.pdf",
        "Chapter 2 - Forest and Wildlife Resources": "chapter2.pdf",
        "Chapter 3 - Water Resources": "chapter3.pdf",
        "Chapter 4 - Agriculture": "chapter4.pdf",
        "Chapter 5 - Mineral and Energy Resources": "chapter5.pdf",
        "Chapter 6 - Manufacturing Industries": "chapter6.pdf"
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
        "Chapter 2 - Algebra": "chapter2.pdf",
        "Chapter 3 - Coordinate Geometry": "chapter3.pdf",
        "Chapter 4 - Geometry": "chapter4.pdf",
        "Chapter 5 - Mensuration": "chapter5.pdf",
        "Chapter 6 - Statistics": "chapter6.pdf",
        "Chapter 7 - Probability": "chapter7.pdf",
        "Chapter 8 - Quadratic Equations": "chapter8.pdf",
        "Chapter 9 - Arithmetic Progressions": "chapter9.pdf",
        "Chapter 10 - Circles": "chapter10.pdf",
        "Chapter 11 - Constructions": "chapter11.pdf",
        "Chapter 12 - Areas Related to Circles": "chapter12.pdf"
    },
    "Science": {
        "Chapter 1 - Matter in Our Surroundings": "chapter1.pdf",
        "Chapter 2 - Is Matter Around Us Pure": "chapter2.pdf",
        "Chapter 3 - Atoms and Molecules": "chapter3.pdf",
        "Chapter 4 - Structure of the Atom": "chapter4.pdf",
        "Chapter 5 - Chemical Reactions and Equations": "chapter5.pdf",
        "Chapter 6 - Acids, Bases and Salts": "chapter6.pdf",
        "Chapter 7 - Metals and Non-Metals": "chapter7.pdf",
        "Chapter 8 - Carbon and Its Compounds": "chapter8.pdf",
        "Chapter 9 - Periodic Classification of Elements": "chapter9.pdf",
        "Chapter 10 - Light": "chapter10.pdf",
        "Chapter 11 - Human Eye and Colourful World": "chapter11.pdf",
        "Chapter 12 - Electricity": "chapter12.pdf"
    }
}

# To maintain chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

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

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a historian with expertise in answering questions related to history. Answer the question as detailed as possible 
    from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, 
    "The answer is not available in the context," and do not provide incorrect information.

    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def display_chat():
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://img.icons8.com/color/48/000000/user.png" width="30" style="margin-right: 10px;">
                    <div>
                        User: {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://img.icons8.com/fluency/48/000000/chatbot.png" width="30" style="margin-right: 10px;">
                    <div>
                        Assistant: {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="PDF Reader", page_icon=":book:")

    st.header("PDF Reader")

    with st.sidebar:
        st.title("Menu:")
        
        # Subject selection
        subject_choice = st.selectbox("Choose a Subject", options=list(subjects.keys()))
        
        # Chapter selection based on the selected subject
        chapters = subjects[subject_choice]
        chapter_choice = st.selectbox("Choose a Chapter", options=list(chapters.keys()))
        
        # Process Chapter button
        if st.button("Process Chapter", key="process_chapter"):
            with st.spinner("Processing..."):
                pdf_path = f"./{subject_choice}/{chapters[chapter_choice]}"
                raw_text = get_pdf_text(pdf_path)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.info("This chatbot uses Google Generative AI model for conversational responses.")
        st.info("Created By Pranav Lejith(Amphibiar)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Submit Question", key="submit_question"):
        if user_question:
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})

            # Get response from the model
            response = user_input(user_question)
            
            # Add model response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display chat history
            display_chat()

if __name__ == "__main__":
    main()
