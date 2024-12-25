import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere.embeddings import CohereEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import cohere
import requests
import re
from seq2seq.seqtoseq import _correct, alphabet
# -------------------------------------------------------
# Load environment variables from .env file
app = Flask(__name__)
api_key = "0DxF4rNKVQskbsF1BHO7VyVj9kSqrka8G7TnLm6l"



def create_loader():
    file_path = "./doc_test.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def faiss_index():
    full_text = create_loader()
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )
     # Split text into chunks
    naive_chunks = text_splitter.split_text(full_text)
    print(naive_chunks)

     # Create a vector store and retriever
    naive_chunk_vectorstore = FAISS.from_texts(
        naive_chunks, 
        embedding=CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")
    )    
    naive_chunk_vectorstore.save_local('E:\\CTU\\luanvan\\test_rag_sematic\\docs_index')

# Initialize RAG Chain on startup
def initialize_rag_chain():
    global naive_chunk_vectorstore
    os.environ["COHERE_API_KEY"] = api_key

    naive_chunk_vectorstore = FAISS.load_local(
        'E:\\CTU\\luanvan\\test_rag_sematic\\docs_index', 
        CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0"),
        allow_dangerous_deserialization=True
    )
    # Define prompt template for retrieval-augmented generation (RAG)
    rag_template = """
    Không trả lời các câu hỏi về chính trị, bạo lực.
    Sử dụng các phần ngữ cảnh sau để trả lời câu hỏi của người dùng. Ưu tiên nội dung bối cảnh đầu tiên. Bạn tên là chatbot CTUMP của trường đại học y dược cần thơ. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Bạn phải trả lời bằng ngôn ngữ của câu hỏi.
    User's Query:
    {question}
    
    Context:
    {context}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    # Initialize Cohere model
    base_model = ChatCohere(model="command-r-08-2024", temperature=0)

    naive_chunk_retriever = naive_chunk_vectorstore.as_retriever()
    naive_chunk_retriever.search_kwargs['fetch_k'] = 30
    naive_chunk_retriever.search_kwargs['maximal_marginal_relevance'] = True
    naive_chunk_retriever.search_kwargs['k'] = 10

    # Define the retrieval-augmented generation chain
    naive_rag_chain = (
        {"context": naive_chunk_retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | base_model
        | StrOutputParser()
    )

    return naive_rag_chain

# Initialize the RAG chain once when the app starts
print("server khởi động lại")
rag_chain = initialize_rag_chain()


@app.route("/")
def index():
    return render_template("index.html")

# URL API của Rasa (thay đổi địa chỉ IP và cổng nếu khác)
RASA_API_URL = "http://localhost:5055/webhook"



@app.route("/check_spell", methods=["POST"])
def check_spell():
    data = request.get_json()
    user_question = data.get("question")  # Sử dụng trường 'question'
  
    if not user_question.strip():
        return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        result = _correct(user_question)

        # Loại bỏ các ký tự không thuộc bảng chữ cái
        text = re.sub(r"[^" + ''.join(alphabet) + ']', '', user_question)
        list_text = text.split()

        response = re.sub(r"[^" + ''.join(alphabet) + ']', '', result)
        list_result = result.split()

        # Kiểm tra độ dài của list_text và list_result
        if len(list_text) != len(list_result):
            print("Dữ liệu sai chính tả nhiều quá")
            
            return jsonify({"error": "Câu hỏi không được sai chính tả."}), 400


        print(f"Câu hỏi ban đầu: {user_question}")
        print(f"Câu đã sửa lỗi: {response}")

        return jsonify({"response": response})  # Sử dụng trường 'response'
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")  # Sử dụng trường 'question'

    print(f"-----------Tin nhắn từ người dùng: {user_question} -----------\n\n")
    # if not user_question.strip():
    #     return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        # response = _correct(user_question)
        # Invoke the RAG chain với câu hỏi của người dùng
        response = rag_chain.invoke(user_question)
        return jsonify({"response": response})  # Sử dụng trường 'response'
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    # Chạy code này khi thêm dữ liệu mới vào txt
    # faiss_index()
    app.run(host='0.0.0.0', port=5000)