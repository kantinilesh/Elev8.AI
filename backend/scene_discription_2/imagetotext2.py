from flask import Flask, request, jsonify, send_file, session
import os
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS
from translate import Translator
from pydub import AudioSegment
import uuid  # For generating unique file names

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Configure upload and audio folders
UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize translator
translator = Translator(to_lang="en")  # Set default target language

def preprocess_image(image):
    """
    Preprocess the image (resize, normalize, etc.) if needed.
    """
    image = image.resize((224, 224))  # Resize to 224x224 (adjust as needed)
    return image

def image_to_base64(image):
    """
    Convert PIL image to base64.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_description(image):
    """
    Send the image to the Groq model and get the description.
    """
    # Convert image to base64
    image_base64 = image_to_base64(image)

    # Call the Groq model
    response = groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

def translate_text(text, dest_language="en"):
    """
    Translate text to the desired language using the translate library.
    """
    translator = Translator(to_lang=dest_language)
    translation = translator.translate(text)
    return translation

def text_to_speech(text, language="en"):
    """
    Convert text to speech using gTTS and save as an audio file with a unique name.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    # Generate a unique filename using UUID
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(app.config["AUDIO_FOLDER"], audio_filename)
    tts.save(audio_path)
    return audio_filename  # Return the unique filename

def convert_to_supported_format(audio_path, output_format="wav"):
    audio = AudioSegment.from_file(audio_path)
    output_path = os.path.splitext(audio_path)[0] + f".{output_format}"
    audio.export(output_path, format=output_format)
    return output_path

@app.route("/")
def home():
    return send_file("templates/imagetotext2.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Handle image upload, generate description, translate, and convert to speech.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and preprocess the image
        image = Image.open(filepath)
        image = preprocess_image(image)

        # Generate description
        description = generate_description(image)

        # Store the description in the session
        session["image_description"] = description

        # Translate description (optional)
        translated_description = translate_text(description, dest_language="en")  # Change language as needed

        # Convert description to speech
        audio_filename = text_to_speech(translated_description, language="en")

        # Return the description and audio file path
        return jsonify({
            "description": translated_description,
            "audio_url": f"/audio/{audio_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handle user questions about the image.
    """
    if "image_description" not in session:
        return jsonify({"error": "No image description found. Please upload an image first."}), 400

    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Get the question and image description
        question = data["question"]
        image_description = session["image_description"]

        # Generate a response using the gemma model
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Use the gemma model
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. The user has uploaded an image with the following description: {image_description}. Answer the user's questions based on this description.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        # Get the response text
        response_text = response.choices[0].message.content

        # Convert response to speech
        audio_filename = text_to_speech(response_text, language="en")

        # Return the response and audio file path
        return jsonify({
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<filename>")
def get_audio(filename):
    """
    Serve the generated audio file.
    """
    return send_file(os.path.join(app.config["AUDIO_FOLDER"], filename))

if __name__ == "__main__":
    app.run(debug=True)