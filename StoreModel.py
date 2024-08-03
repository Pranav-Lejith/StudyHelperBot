import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Path to the root directory containing subject folders
root_directory = "./"

# Dictionary containing the subject and chapter details
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

# Single API Key (replace with your actual key)
api_key = "AIzaSyCZgqu-GCLTD1L3ni5nI6x2kRVhDqELB1k"

def get_current_api_key():
    return api_key

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

def preprocess_pdfs_and_save_vector_store():
    all_texts = []

    for subject, chapters in subjects.items():
        subject_directory = os.path.join(root_directory, subject)
        for chapter_name, pdf_filename in chapters.items():
            pdf_path = os.path.join(subject_directory, pdf_filename)
            text = get_pdf_text(pdf_path)
            text_chunks = get_text_chunks(text)
            all_texts.extend(text_chunks)

    # Create and save vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_current_api_key())
    vector_store = FAISS.from_texts(all_texts, embedding=embeddings)
    vector_store.save_local("faiss_index")

if __name__ == "__main__":
    preprocess_pdfs_and_save_vector_store()
