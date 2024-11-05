import os
import glob
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
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Replace with your actual API key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Create FAISS vector database from documents
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Define RAG prompt templates for different recommendations
def generate_treatment_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan untuk memberikan rekomendasi pengobatan kepada pasien berdasarkan informasi yang tersedia.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pengobatan yang singkat namun spesifik dan jelas meliputi:
    1. Obat yang disarankan beserta dosisnya (jika mungkin).
    2. Metode pengobatan yang sesuai.
    3. Langkah perawatan yang harus dilakukan oleh petugas medis terhadap pasien.
    """
    return prompt

def generate_lifestyle_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang membantu petugas kesehatan memberikan rekomendasi pola hidup sehat kepada pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi pola hidup yang singkat namun spesifik dan jelas yang mencakup:
    1. Aktivitas fisik yang aman dan direkomendasikan (misalnya, jenis olahraga dan frekuensinya).
    2. Pola makan dan jenis makanan yang sebaiknya dikonsumsi dan dihindari (contoh: makanan yang meningkatkan imunitas).
    3. Kebiasaan sehari-hari yang dapat membantu pemulihan, termasuk tips manajemen stres dan tidur.
    4. Instruksi khusus untuk menjaga kebersihan dan mencegah penularan.
    """
    return prompt

def generate_followup_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan yang memberikan rekomendasi penanganan lanjutan bagi petugas kesehatan untuk pasien yang terindikasi TB atau tidak dengan informasi berikut.

    **Profil dan Riwayat Pasien**:
    {query}

    **Riwayat Medis dan Keterangan Medis**:
    {context}

    Berdasarkan informasi di atas, berikan rekomendasi penanganan lanjutan yang singkat namun spesifik dan jelas mencakup:
    1. Jadwal kontrol kesehatan atau pemeriksaan lanjutan yang disarankan.
    2. Pengujian tambahan atau pemeriksaan yang mungkin diperlukan (contoh: X-ray atau tes laboratorium).
    3. Tanda atau gejala yang perlu diwaspadai sebagai indikasi komplikasi.
    4. Saran untuk pemulihan yang berkelanjutan, seperti adaptasi pola hidup, manajemen stres, dan dukungan sosial yang dibutuhkan.
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
            prompt = generate_treatment_prompt(query=query, context=context)
        elif recommendation_type == "Rekomendasi Pola Hidup":
            prompt = generate_lifestyle_prompt(query=query, context=context)
        elif recommendation_type == "Rekomendasi Penanganan Lanjutan":
            prompt = generate_followup_prompt(query=query, context=context)

        # Buat pesan HumanMessage dan dapatkan hasil dari model LLM
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)
        st.markdown(answer.content)

if __name__ == "__main__":
    main()
