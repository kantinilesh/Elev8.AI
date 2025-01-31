from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from groq import Groq
from phi.assistant import Assistant
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Load Groq API key from .env
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Knowledge Base and Storage
knowledge_base = None
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

def truncate_context(context: str, max_tokens: int = 10000) -> str:
    """
    Truncate the context to a maximum number of tokens to avoid exceeding rate limits.
    """
    tokens = context.split()  # Split context into tokens (words)
    truncated_tokens = tokens[:max_tokens]  # Take the first `max_tokens` tokens
    return " ".join(truncated_tokens)  # Join tokens back into a string

def ask_groq(question: str, context: str) -> str:
    """
    Function to query the Groq model for questions, restricted to the provided context (PDF content).
    """
    try:
        # Truncate the context to avoid exceeding token limits
        truncated_context = truncate_context(context, max_tokens=10000)
        prompt = f"Context: {truncated_context}\n\nQuestion: {question}\n\nAnswer the question based only on the context provided. Format the response using Markdown for proper spacing, paragraphs, and bold headings."
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",  # Use the appropriate Groq model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

def clear_vector_db(collection_name: str):
    """
    Clear all embeddings from the vector database for a specific collection.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Execute a raw SQL query to delete all rows from the table
        with engine.connect() as connection:
            query = text("DELETE FROM items")  # Remove the condition
            connection.execute(query)
            connection.commit()
        print(f"Cleared all embeddings from items table")
    except Exception as e:
        print(f"Error clearing vector database: {str(e)}")
        raise

@app.route("/")
def index():
    return render_template("talktopdf.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Clear the existing vector database before loading a new PDF
    collection_name = "items"
    clear_vector_db(collection_name)

    # Load the PDF into the knowledge base
    global knowledge_base
    knowledge_base = PDFKnowledgeBase(
        path=file_path,
        vector_db=PgVector2(collection=collection_name, db_url=db_url),
        reader=PDFReader(chunk=True),
    )
    knowledge_base.load()

    return jsonify({"message": "File uploaded successfully", "filename": filename})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not knowledge_base:
        return jsonify({"error": "No PDF uploaded"}), 400

    # Search the PDF knowledge base for relevant context
    context_results = knowledge_base.search(question)
    if not context_results:
        return jsonify({"error": "No relevant information found in the PDF"}), 404

    # Extract the `content` attribute from each Document object
    context = " ".join([doc.content for doc in context_results])

    # Use Groq to answer the question based on the PDF context
    groq_response = ask_groq(question, context)

    # If the user asks for important questions, generate them using Groq
    if "important questions" in question.lower():
        prompt = f"Context: {context}\n\nGenerate 5 important questions based on the provided context."
        important_questions = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
        )
        groq_response = important_questions.choices[0].message.content

    return jsonify({"response": groq_response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3006,debug=True)