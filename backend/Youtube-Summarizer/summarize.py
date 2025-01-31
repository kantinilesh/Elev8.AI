from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Verify API key is present
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Define the prompt template
prompt = """You are a Youtube video summarizer. Summarize the following transcript in an easy-to-understand way, 
providing key points within 250 words. Focus on the main ideas and important details. Format the summary into clear 
bullet points for better readability. Use **bold** for headings to highlight key sections. Do not hallucinate.

Transcript:
{transcript}

Please provide a clear and concise summary in bullet points with headings in **bold**."""

def extract_transcript_details(youtube_video_url):
    """Extract transcript from YouTube video"""
    try:
        # Handle different YouTube URL formats
        if "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1]
        elif "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
            
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(item["text"] for item in transcript_data)
        return transcript.strip(), video_id
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None, None

def generate_groq_summary(transcript_text):
    """Generate summary using Groq API via LangChain"""
    try:
        # Initialize the ChatGroq model
        model = ChatGroq(
            model_name="llama-3.1-8b-instant",  # Using Llama3 model
            groq_api_key=groq_api_key,
            temperature=0.7,
            max_tokens=4000
        )
        
        # Format the prompt with the transcript
        formatted_prompt = prompt.format(transcript=transcript_text)
        
        # Generate the summary
        response = model.invoke(formatted_prompt)
        
        # Extract and return the summary text
        if response and hasattr(response, 'content'):
            return response.content.strip()
        else:
            print("No response content found")
            return None
            
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get the YouTube URL from the request
    data = request.json
    youtube_link = data.get("youtube_link")
    
    if not youtube_link:
        return jsonify({"error": "YouTube link is required"}), 400
    
    try:
        # Extract transcript
        transcript_text, video_id = extract_transcript_details(youtube_link)
        
        if not transcript_text or not video_id:
            return jsonify({"error": "Failed to extract transcript"}), 400
        
        # Generate summary
        summary = generate_groq_summary(transcript_text)
        
        if not summary:
            return jsonify({"error": "Failed to generate summary"}), 500
        
        # Return the result as JSON
        return jsonify({
            "summary": summary,
            "thumbnail_url": f"http://img.youtube.com/vi/{video_id}/0.jpg"
        })
        
    except Exception as e:
        print(f"Error in summarize route: {e}")
        return jsonify({"error": str(e)}), 500

# New route for YouTube Summarizer form
@app.route("/youtube-summarizer")
def youtube_summarizer():
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)