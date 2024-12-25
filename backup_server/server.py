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
# Load environment variables from .env file
app = Flask(__name__)
api_key = "YjuQlm1x2x957ejDpCtmSCIAmpzafWS2hhBBycVk"



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
    Sử dụng các phần ngữ cảnh sau để trả lời câu hỏi của người dùng. Ưu tiên nội dung bối cảnh đầu tiên. Bạn là chatbot của trường đại học y dược cần thơ. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Bạn phải trả lời bằng ngôn ngữ của câu hỏi.
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


# # API nhận tin nhắn từ người dùng và phản hồi
# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.get_json()
#     user_question = data.get("question")  # Sử dụng trường 'question'
#     print(f"question: {user_question}")
#     if not user_question.strip():
#         return jsonify({"error": "Câu hỏi không được để trống."}), 400
    
#     try:
#         # Gửi yêu cầu HTTP POST tới Rasa
#         # Dữ liệu cần gửi tới Rasa
#         post_data = {"sender": "user", "message": user_question}
#         rasa_response = requests.post(RASA_API_URL, json=post_data)
#         rasa_response.raise_for_status()

#         # Xử lý phản hồi từ Rasa
#         if rasa_response.status_code == 200:
#             rasa_data = rasa_response.json()
#             if rasa_data:
#                 # Phản hồi từ Rasa
#                 bot_reply = rasa_data[0].get('text', 'Xin lỗi, tôi không hiểu câu hỏi của bạn.')
#                 # print(f"PHẢN HỒI TỪ RASA : {bot_reply}")
#             else:
#                 bot_reply = 'Không nhận được phản hồi từ Rasa.'
#         else:
#             bot_reply = 'Lỗi kết nối với Rasa.'
#     except requests.RequestException as e:
#         bot_reply = f'Lỗi kết nối với Rasa: {e}'
#           # rag = retrieval(user_ques,bot_reply)

#     return jsonify({
#             "response": bot_reply
#         })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")  # Sử dụng trường 'question'
    print(f"Received question: {user_question}")
    if not user_question.strip():
        return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        
        # Invoke the RAG chain với câu hỏi của người dùng
        response = rag_chain.invoke(user_question)
        return jsonify({"response": response})  # Sử dụng trường 'response'
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":

    # Chạy code này khi thêm dữ liệu mới vào txt
    # faiss_index()

    app.run(host='0.0.0.0', port=5000)