import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def main():
    # ... (前面的程式碼都一樣) ...
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY") is None:
        print("錯誤：找不到 GOOGLE_API_KEY。請檢查你的 .env 檔案。")
        return
    print("Google API Key 載入成功！")
    
    persist_directory = 'chroma_db'
    print("正在載入向量資料庫...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("向量資料庫載入完成！")
    
    print("正在建立問答鏈...")
    
    # --- ✨✨✨ 修改處：換成更輕量的 Flash 模型 ✨✨✨ ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    # --------------------------------------------------
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    print("問答鏈建立完成！現在你可以開始提問了。")

    # ... (後面的問答迴圈都一樣) ...
    while True:
        query = input("\n請輸入你的問題 (或輸入 'exit' 結束): ")
        if query.lower() == 'exit':
            break
        
        print("思考中...")
        try:
            result = qa_chain.invoke(query)
            print("\n答案:")
            print(result["result"])
        except Exception as e:
            print(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()