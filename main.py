import streamlit as st
import os
from rag import pdf_read, get_chunks, vector_store, check_database_exists
from agent import get_answer_with_rag

# --------------------------------------------------
# 读取多个债券文件
# --------------------------------------------------
def read_multiple_bond_files(files):
    if not files:
        return ""

    all_contents = []

    for file in files:
        ext = file.name.split(".")[-1].lower()

        # PDF
        if ext == "pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                p = page.extract_text()
                if p:
                    text += p
            all_contents.append(f"### File: {file.name}\n{text}")

        # TXT
        elif ext == "txt":
            text = file.read().decode("utf-8", errors="ignore")
            all_contents.append(f"### File: {file.name}\n{text}")

        # DOCX
        elif ext == "docx":
            from docx import Document
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            all_contents.append(f"### File: {file.name}\n{text}")

        # CSV
        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(file)
            all_contents.append(f"### File: {file.name}\n{df.to_string()}")

    return "\n\n".join(all_contents)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config("Bond Analysis Assistant", layout="wide")
    st.header("💼 Bond Analysis Assistant (Multi-Bond Comparison Enabled)")

    st.markdown(
        "Upload multiple bond documents such as term sheets, prospectuses, or basic info files, "
        "and ask any analytical question. AI will compare, analyze, and summarize differences."
    )

    # --------------------------------------------------
    # 用户提问
    # --------------------------------------------------
    user_question = st.text_input(
        "💬 Your Question About the Bonds:",
        placeholder="e.g., Compare the credit risk and duration of the uploaded bonds.",
        disabled=not check_database_exists()
    )

    # 用户输入的补充信息
    bond_text = st.text_area(
        "📝 (Optional) Add Notes or Manual Bond Terms:",
        placeholder="e.g., Bond A: Callable 2027. Bond B: Higher coupon...",
        height=180,
        disabled=not check_database_exists()
    )

    # --------------------------------------------------
    # 多债券文件上传
    # --------------------------------------------------
    st.subheader("📄 Upload Bond Information Files (Multiple Allowed)")
    bond_files = st.file_uploader(
        "Upload PDF / DOCX / TXT / CSV files for analysis and comparison:",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
        disabled=not check_database_exists()
    )

    if bond_files:
        st.info(f"📚 {len(bond_files)} bond files uploaded:")
        for f in bond_files:
            st.write(f"- {f.name}")

    # --------------------------------------------------
    # 提交分析按钮
    # --------------------------------------------------
    if st.button("🚀 Analyze Bonds", disabled=not check_database_exists()):
        if not user_question:
            st.error("❌ Please enter a question.")
            return

        uploaded_contents = read_multiple_bond_files(bond_files)

        merged_bond_info = (
            "### Bond Documents Provided:\n\n"
            + (uploaded_contents or "No bond files uploaded.\n")
            + "\n\n### User Notes:\n"
            + (bond_text or "")
        )

        with st.spinner("🤖 DeepSeek is analyzing the bonds..."):
            response = get_answer_with_rag(user_question, merged_bond_info)

        st.write("### 📘 AI Analysis Result:")
        st.write(response["output"])

    # --------------------------------------------------
    # 侧边栏：构建知识库
    # --------------------------------------------------
    with st.sidebar:
        st.title("📁 Knowledge Base Management")

        if check_database_exists():
            st.success("✅ Knowledge Base Status: Ready")
        else:
            st.info("📝 Status: Please upload bond-related PDFs to build the knowledge base.")

        # 清空数据库按钮
        if st.button("🗑️ Clear Knowledge Base"):
            import shutil
            try:
                if os.path.exists("faiss_db_bond"):
                    shutil.rmtree("faiss_db_bond")
                st.success("Knowledge base cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")

        st.markdown("---")

        # 上传 PDF 构建 RAG 知识库
        pdf_doc = st.file_uploader(
            "📎 Upload Bond-related PDFs for Knowledge Base",
            accept_multiple_files=True,
            type=['pdf']
        )

        if pdf_doc:
            st.info(f"📄 {len(pdf_doc)} PDFs selected")

        if st.button("⚙️ Build Knowledge Base", disabled=not pdf_doc):
            with st.spinner("🔧 Processing PDFs..."):
                try:
                    raw_text = pdf_read(pdf_doc)
                    chunks = get_chunks(raw_text)
                    vector_store(chunks)
                    st.success("🎉 Knowledge base created successfully!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building knowledge base: {e}")


if __name__ == "__main__":
    main()
