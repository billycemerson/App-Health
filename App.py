import os
import glob
import signal
import sys
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage

@st.cache_resource
def init():
    # Load all PDFs from the specified folder
    pdf_folder_path = "./data/"
    all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    # Load each PDF document and split text
    documents = []
    for pdf_path in all_pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(pdf_docs))
    
    print(f"Total loaded document chunks: {len(documents)}")
    
    # Set up embeddings and LLM with Google Gemini API
    GEMINI_API_KEY = "AIzaSyDFQrUxPXyeVGU66oxymNMeK9IZy_Z272U"  # Replace with your actual API key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Create FAISS vector database from documents
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Define RAG prompt templates for different recommendations
def generate_prompt(query, context, recommendation_type):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan {recommendation_type} kepada pasien berdasarkan informasi pasien yang ada.

    Informasi Pasien:
    {query}

    Keterangan Medis:
    {context}

    Berikan rekomendasi {recommendation_type} yang dapat diterapkan sesuai kondisi pasien yang ada. Pastikan rekomendasi spesifik dan relevan.
    """
    return prompt

# Streamlit App
def main():
    llm, retriever = init()
    st.title("Sistem Rekomendasi Kesehatan Berbasis RAG")

    # Input fields
    profil_pasien = st.text_input("Masukkan Profil Pasien (umur, jenis kelamin, dll):")
    riwayat_pasien = st.text_area("Masukkan Riwayat Pasien:")
    pola_hidup = st.text_area("Masukkan Pola Hidup Pasien:")
    hasil_ctscan = st.selectbox("Masukkan Hasil CT Scan", ("TB", "Tidak TB"))

    # Recommendation Type Selection
    recommendation_type = st.radio("Pilih Jenis Rekomendasi:",
                                    ("Rekomendasi Pengobatan",
                                     "Rekomendasi Pola Hidup",
                                     "Rekomendasi Penanganan Lanjutan"))

    if st.button("Dapatkan Rekomendasi"):
        # Gabungkan data pasien sebagai query untuk konteks rekomendasi
        query = f"Profil pasien: {profil_pasien}. Riwayat: {riwayat_pasien}. Pola Hidup:{pola_hidup}. Hasil CT Scan: {hasil_ctscan}."
        context = "\n".join([result.page_content for result in retriever.get_relevant_documents(query)])

        # Generate prompt based on user selection
        if recommendation_type == "Rekomendasi Pengobatan":
            prompt = generate_prompt(query=f"{query} Rekomendasi pengobatan untuk petugas kesehatan terhadap pasien TB.",
                                     context=context,
                                     recommendation_type="Rekomendasi Pengobatan")
        elif recommendation_type == "Rekomendasi Pola Hidup":
            prompt = generate_prompt(query=f"{query} Rekomendasi pola hidup untuk pasien TB.",
                                     context=context,
                                     recommendation_type="Rekomendasi Pola Hidup")
        elif recommendation_type == "Rekomendasi Penanganan Lanjutan":
            prompt = generate_prompt(query=f"{query} Rekomendasi penanganan lanjutan untuk pasien TB.",
                                     context=context,
                                     recommendation_type="Rekomendasi Penanganan Lanjutan")

        # Buat pesan HumanMessage dan dapatkan hasil dari model LLM
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        st.markdown(answer.content)
        # st.text_area("Jawaban:", write_message(answer), height=300)

# def write_message(answer):
#     return st.markdown(answer.content)

if __name__ == "__main__":
    main()
