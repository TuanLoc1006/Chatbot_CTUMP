import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere.embeddings import CohereEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import cohere

# Load environment variables from .env file

api_key = "YjuQlm1x2x957ejDpCtmSCIAmpzafWS2hhBBycVk"

app = Flask(__name__)

# Initialize RAG Chain on startup
def initialize_rag_chain():
    # Read the .txt file with UTF-8 encoding
    with open("./docs.txt", "r", encoding="utf-8") as f:
        alice_in_wonderland = f.read()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )

    # API key for Cohere from environment variable
    
    os.environ["COHERE_API_KEY"] = api_key

    # Split text into chunks
    naive_chunks = text_splitter.split_text(alice_in_wonderland)

    # Initialize Cohere embeddings and chunker
    cohere_embeddings = CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")
    semantic_chunker = SemanticChunker(cohere_embeddings, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents([alice_in_wonderland])

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
    cohere_client = cohere.Client(api_key)
    base_model = ChatCohere(model="command-r-08-2024", temperature=0)

    # Create a vector store and retriever
    naive_chunk_vectorstore = FAISS.from_texts(
        naive_chunks, 
        embedding=CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")
    )
    naive_chunk_retriever = naive_chunk_vectorstore.as_retriever(search_kwargs={"k": 15})

    # Define the retrieval-augmented generation chain
    naive_rag_chain = (
        {"context": naive_chunk_retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | base_model
        | StrOutputParser()
    )

    return naive_rag_chain

# Initialize the RAG chain once when the app starts
rag_chain = initialize_rag_chain()

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")  # Sử dụng trường 'question'
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
    app.run(debug=True)