__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

@st.cache_resource
def load_chain():
    """
    載入所有必要的模型和 RAG 鏈。
    """
    print("--- 正在載入模型與 RAG 鏈... ---")
    
    load_dotenv()
    persist_directory = 'chroma_db'
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(f"錯誤：找不到向量資料庫！\n請確認 'chroma_db' 資料夾存在於您執行指令的目錄：{os.getcwd()}")
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("錯誤：找不到 GOOGLE_API_KEY。\n請檢查你的 .env 檔案是否已正確設定。")
        
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # --- ✨✨✨ 修改處 1 ✨✨✨ ---
    # 我們在這裡加入 return_source_documents=True
    # 讓 RAG 鏈回傳它參考的原始文件
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # <-- 新增這一行
    )
    # -----------------------------
    print("--- 模型與 RAG 鏈載入完畢 ---")
    return qa_chain

def main():
    """
    Streamlit Web App 的主體。
    """
    st.set_page_config(page_title="AI 財報分析師", page_icon="📈")
    st.title("📈 AI 財報分析師")

    try:
        qa_chain = load_chain()
    except Exception as e:
        st.error(f"應用程式啟動失敗：\n{e}")
        st.stop() 

    st.markdown("你好！我是你的 AI 財報分析助理。我已經讀完了台積電 (TSMC) 2023 年的 20-F 年報，你可以問我任何相關問題。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # --- ✨✨✨ 修改處 3 ✨✨✨ ---
            # 如果訊息是 AI 的，且包含來源，就顯示來源
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("查看引用來源"):
                    for source in message["sources"]:
                        st.markdown(f"**來源頁數：{source['page']}**")
                        st.markdown(f"> {source['content']}")
            # -----------------------------

    if prompt := st.chat_input("你想問什麼？ (例如：台積電的主要風險是什麼？)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("思考中..."):
            try:
                # --- ✨✨✨ 修改處 2 ✨✨✨ ---
                # .invoke 現在會回傳一個包含 'result' 和 'source_documents' 的字典
                result = qa_chain.invoke(prompt)
                response_text = result["result"]
                
                # 取得來源文件並處理
                sources = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        # PyPDFLoader 的頁碼是從 0 開始，我們 +1 讓它更易讀
                        page_num = doc.metadata.get('page', 'N/A')
                        if page_num != 'N/A':
                            page_num += 1
                        
                        sources.append({
                            "content": doc.page_content,
                            "page": page_num
                        })
                
                # 顯示 AI 回應
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    # 在 AI 回應下方，立即顯示來源
                    if sources:
                        with st.expander("查看引用來源"):
                            for source in sources:
                                st.markdown(f"**來源頁數：{source['page']}**")
                                st.markdown(f"> {source['content']}")
                
                # 將 AI 回應與來源一起存入聊天紀錄
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources  # <-- 將來源也存起來
                })
                # -----------------------------
                
            except Exception as e:
                error_message = f"發生錯誤：{e} \n\n(這很可能是因為 Google API 的免費額度達到了速率限制，請稍候一分鐘再試。)"
                with st.chat_message("assistant"):
                    st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":

    main()
