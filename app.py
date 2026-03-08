import streamlit as st
import os
import time
from dotenv import load_dotenv
from pypdf import PdfReader 

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Environment
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# กำหนดพาธของฐานข้อมูลให้ชัดเจน
DB_PATH = os.path.join(os.getcwd(), "faiss_db")

st.set_page_config(page_title="Automotive Tech-Copilot Pro", layout="wide", page_icon="🚗")
st.title("🚗 Automotive Tech-Copilot Pro")
st.subheader("ระบบวิเคราะห์ปัญหาทางเทคนิคด้วย AI (Root Cause Analysis Support)")

# 2. ฟังก์ชันประมวลผลไฟล์ PDF (Batch Processing)
def get_vector_store(pdf_docs):
    if not pdf_docs:
        return None
    
    all_text = ""
    with st.spinner("กำลังอ่านข้อมูลจากไฟล์ PDF..."):
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                all_text += page.extract_text() or ""
    
    # แบ่งข้อมูลเป็นก้อนเพื่อทำ Indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    chunks = text_splitter.split_text(all_text)
    
    # ใช้ Embedding Model มาตรฐาน
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    with st.status("📦 กำลังสร้างฐานข้อมูลความรู้ทางเทคนิค...", expanded=True) as status:
        vector_store = None
        batch_size = 5 
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    vector_store.add_texts(batch)
                
                vector_store.save_local(DB_PATH)
                time.sleep(2) # พักเล็กน้อยป้องกัน Rate Limit
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
                break
            
        status.update(label="🚀 AI พร้อมวิเคราะห์ข้อมูลแล้วครับ", state="complete", expanded=False)
    return vector_store

# 3. Prompt Engineering (ปรับให้เป็นกลางและเป็นมืออาชีพ)
prompt_template = ChatPromptTemplate.from_template("""
คุณคือ Automotive Technical Assistant ผู้เชี่ยวชาญด้านวิศวกรรมยานยนต์สมัยใหม่
จงวิเคราะห์ปัญหาจากคู่มือเทคนิคที่ให้มาโดยใช้หลัก Root Cause Analysis (RCA) 
อธิบายขั้นตอนให้ชัดเจน เข้าใจง่าย สำหรับช่างเทคนิคและที่ปรึกษาบริการ

Context: {context}
Question: {input}

คำตอบที่วิเคราะห์ตามหลักเหตุและผล:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Streamlit UI (Sidebar)
with st.sidebar:
    st.header("🔧 คลังความรู้เทคนิค")
    pdf_docs = st.file_uploader("อัปโหลดไฟล์คู่มือเทคนิค (PDF Public)", accept_multiple_files=True)
    if st.button("Train AI on Documents"):
        if pdf_docs:
            get_vector_store(pdf_docs)
            st.success("สร้างฐานข้อมูลสำเร็จ!")
        else:
            st.warning("กรุณาอัปโหลดไฟล์ PDF ก่อนครับ")

# 5. ส่วนการรับคำถาม
user_question = st.text_input("สอบถามปัญหาเทคนิคหรือขั้นตอนการตรวจสอบ:")

if user_question:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if not os.path.exists(DB_PATH):
        st.error("ไม่พบฐานข้อมูล กรุณาอัปโหลดไฟล์และกด Train AI ก่อนครับ")
    else:
        try:
            # โหลดฐานข้อมูลที่สร้างไว้
            new_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            
            # ใช้โมเดล Gemini 2.5 Flash ตัวล่าสุดที่คุณเจอ
            llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
            
            rag_chain = (
                {"context": new_db.as_retriever() | format_docs, "input": RunnablePassthrough()} 
                | prompt_template 
                | llm 
                | StrOutputParser()
            )
            
            with st.spinner("กำลังวิเคราะห์คำตอบจากคู่มือ..."):
                response = rag_chain.invoke(user_question)
                st.markdown("---")
                st.markdown("### 🛠 ผลการวิเคราะห์:")
                st.write(response)
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")