import os
import glob
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage

# Initialize the model and retriever
@st.cache_resource
def init():
    pdf_folder_path = "./data/"
    all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    documents = []
    for pdf_path in all_pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(pdf_docs))
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="YOUR_GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="YOUR_GEMINI_API_KEY")
    
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Generate prompt
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

# Main app function
def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Pilih Menu", ["Home", "Prediksi", "Sistem Rekomendasi", "About Us"])

    if menu == "Home":
        st.title("Selamat Datang di Sistem Rekomendasi Kesehatan TB")
        st.write("Sistem ini dirancang untuk membantu petugas kesehatan dalam memberikan rekomendasi yang tepat bagi pasien dengan penyakit Tuberkulosis (TB).")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Mycobacterium_tuberculosis.jpg/1200px-Mycobacterium_tuberculosis.jpg", caption="Bakteri Mycobacterium tuberculosis - Penyebab TB")
        st.write("""
        ### Apa itu Tuberkulosis (TB)?
        Tuberkulosis adalah penyakit infeksi menular yang disebabkan oleh bakteri Mycobacterium tuberculosis. Penyakit ini terutama menyerang paru-paru, tetapi juga dapat menyebar ke bagian tubuh lainnya.
        
        ### Gejala TB
        - Batuk berlangsung lama (lebih dari 3 minggu)
        - Demam, terutama pada malam hari
        - Kehilangan berat badan
        - Keringat berlebih di malam hari
        """)
        
    elif menu == "Prediksi":
        st.title("Prediksi TB")
        st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena penyakit TB.")
        
        # Input fields for prediction
        umur = st.number_input("Umur", min_value=0, max_value=120)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        riwayat_keluarga = st.selectbox("Apakah ada riwayat keluarga dengan TB?", ["Ya", "Tidak"])
        pola_hidup = st.selectbox("Pola hidup pasien (merokok, alkohol, dll)", ["Baik", "Buruk"])
        st.button("Prediksi")

        st.write("**Catatan:** Model prediksi akan dikembangkan di masa mendatang.")
        
    elif menu == "Sistem Rekomendasi":
        llm, retriever = init()
        st.title("Sistem Rekomendasi Kesehatan TB")

        # Input fields for recommendation system
        profil_pasien = st.text_input("Masukkan Profil Pasien (umur, jenis kelamin, dll):")
        riwayat_pasien = st.text_area("Masukkan Riwayat Pasien:")
        pola_hidup = st.text_area("Masukkan Pola Hidup Pasien:")
        hasil_ctscan = st.selectbox("Masukkan Hasil CT Scan", ("TB", "Tidak TB"))

        recommendation_type = st.radio("Pilih Jenis Rekomendasi:", ["Rekomendasi Pengobatan", "Rekomendasi Pola Hidup", "Rekomendasi Penanganan Lanjutan"])

        if st.button("Dapatkan Rekomendasi"):
            query = f"Profil pasien: {profil_pasien}. Riwayat: {riwayat_pasien}. Pola Hidup:{pola_hidup}. Hasil CT Scan: {hasil_ctscan}."
            context = "\n".join([result.page_content for result in retriever.get_relevant_documents(query)])

            if recommendation_type == "Rekomendasi Pengobatan":
                prompt = generate_prompt(query, context, "Rekomendasi Pengobatan")
            elif recommendation_type == "Rekomendasi Pola Hidup":
                prompt = generate_prompt(query, context, "Rekomendasi Pola Hidup")
            else:
                prompt = generate_prompt(query, context, "Rekomendasi Penanganan Lanjutan")

            messages = [HumanMessage(content=prompt)]
            answer = llm(messages=messages)
            st.markdown(answer.content)

    elif menu == "About Us":
        st.title("Tentang Kami")
        st.write("Sistem ini dikembangkan oleh tim kami untuk membantu penanganan pasien TB dengan pendekatan berbasis AI.")
        
        # Display team members
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image("https://via.placeholder.com/150", width=100, caption="Nama 1")
            st.write("ML Engineer")
        
        with col2:
            st.image("https://via.placeholder.com/150", width=100, caption="Nama 2")
            st.write("AI Researcher")
        
        with col3:
            st.image("https://via.placeholder.com/150", width=100, caption="Nama 3")
            st.write("Data Scientist")
        
        with col4:
            st.image("https://via.placeholder.com/150", width=100, caption="Nama 4")
            st.write("Software Engineer")
            
if __name__ == "__main__":
    main()
