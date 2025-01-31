from flask import Flask, request, jsonify, render_template
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.wikipedia import WikipediaTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import time
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Initialize Flask app
app = Flask(__name__)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Initialize Groq LLM
groq_model = Groq(id="llama-3.3-70b-versatile")
# Initialize Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for relevant information.",
    llm=groq_model,
    tools=[DuckDuckGo()],
    instructions=[
        "Always include credible sources.",
        "Focus on information that helps guide career decisions.",
    ],
    show_tool_calls=True,
    markdown=True
)

# Initialize Interest-Based Educator Agent
interest_agent = Agent(
    name="Interest-Based Educator Agent",
    role="Provide career advice and roadmap based on user interests.",
    llm=groq_model,
    tools=[WikipediaTools()],
    instructions=[
        "Provide a clear and structured text-based roadmap.",
        "Use **bold headings**, bullet points, and numbered lists for better readability.",
        "Suggest career fields aligned with the user's interests.",
        "Include educational requirements, key skills, and career progression.",
    ],
    show_tool_calls=True,
    markdown=True
)

# Initialize Multi-Agent System
multi_ai_agent = Agent(
    team=[web_search_agent, interest_agent],
    instructions=[
        "Analyze the input dictionary to identify key interests and align them with career options.",
        "Provide career suggestions in a structured format with actionable roadmaps.",
        "Include sources for credibility wherever applicable."
    ],
    show_tool_calls=True,
    markdown=True
)

def get_response_text(response) -> str:
    """Extract text content from RunResponse object or string"""
    if hasattr(response, 'content'):
        return str(response.content)
    return str(response)

@app.route("/")
def home():
    return render_template("roadmap.html")

@app.route("/generate-roadmap", methods=["POST"])
def generate_roadmap():
    try:
        # Get the answers from the request
        answers = request.json

        # Convert answers to the interest_list format
        interest_list = {
            "What subjects do you enjoy the most?": answers.get("q1", ""),
            "How do you feel about problem-solving tasks or logical challenges (Rate from 1-10)?": answers.get("q2", ""),
            "Do you enjoy tasks that involve creativity, such as designing or storytelling?": answers.get("q3", ""),
            "How comfortable are you with working on numbers, statistics, or data analysis? (Rate from 1-10)": answers.get("q4", ""),
            "Do you prefer working independently or as part of a team?": answers.get("q5", ""),
            "Would you rather work on hands-on projects (e.g., robotics) or conceptual work (e.g., research)?": answers.get("q6", ""),
            "Which of these activities excites you the most?": answers.get("q7", ""),
            "What kind of impact do you want to create through your work?": answers.get("q8", ""),
            "Which career paths or technologies interest you?": answers.get("q9", ""),
            "What hobbies or activities do you pursue in your free time?": answers.get("q10", "")
        }

        # Generate the roadmap (use your existing logic)
        prompt = """
Based on the following interests, suggest 2-3 suitable career options. For each career path, provide:
1. **Educational Requirements**: List specific degrees, certifications, and courses needed.
2. **Key Skills**: List 4-5 essential skills required for success.
3. **Career Progression**: Outline 3-4 levels of career advancement.
4. **Clear Milestones**: Define specific achievements needed at each stage.

Present the information in a structured format with **bold headings**, bullet points, and numbered lists for better readability.

Interests:
""" + "\n".join(f"{key}: {value}" for key, value in interest_list.items())

        response = multi_ai_agent.run(prompt, markdown=True)
        roadmap_text = get_response_text(response)

        # Return the roadmap text
        return jsonify({
            "roadmap_text": roadmap_text
        })

    except Exception as e:
        print(f"Error generating roadmap: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3019, debug=True)