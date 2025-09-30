A sophisticated, self-correcting Retrieval-Augmented Generation (RAG) agent built with LangGraph and Streamlit. CogniGraph goes beyond standard RAG by intelligently validating its own retrieved information, enabling a fallback mechanism to ensure the most accurate and contextually relevant answers.

**Key Features**
**Self-Correction Loop**: Autonomously grades the relevance of retrieved documents. If the context is insufficient, it rewrites the user's query for a more effective web search.

Graph-Based Architecture: Utilizes LangGraph to create a robust, stateful agent that visualizes its own thought process (retrieve → grade → decide → rewrite → search → generate).

**Hybrid Information Retrieval**: Seamlessly combines a local FAISS vector store with real-time web search capabilities via the Tavily API.

**Interactive UI**: A polished, chat-based web interface built with Streamlit that shows the agent's internal steps and streams the final answer.

CI/CD Pipeline: Includes a GitHub Actions workflow for automated testing and deployment to Streamlit Community Cloud.

**Architecture:** The CogniGraph Agent Flow
The agent operates as a state machine, moving through a series of nodes to process a query. This graph-based approach allows for complex, cyclical reasoning that is not possible in a simple chain.

(Note: Replace the tag above with an actual image of your app.get_graph().draw_mermaid_png() output for a visual representation in your repository.)

**🛠️ Technology Stack**
Backend & Orchestration: Python, LangChain, LangGraph

Frontend: Streamlit

LLMs & Embeddings: OpenAI (GPT-3.5)

Vector Store: FAISS (for local document indexing)

Tools: Tavily Search API (for real-time web search)

CI/CD: GitHub Actions

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.9+

An OpenAI API Key

A Tavily API Key

2. Clone the Repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>

3. Set Up Environment Variables
Create a file named .env in the root of your project directory. This file is used to store your API keys securely.

# .env
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."

Replace "sk-..." and "tvly-..." with your actual API keys.

4. Install Dependencies
It is recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt

5. Run the Streamlit Application
Once the dependencies are installed and your environment variables are set, you can start the application.

streamlit run app.py

Your web browser should automatically open to the application's URL (usually http://localhost:8501).

**⚙️ CI/CD Pipeline**
This repository is configured with a GitHub Actions workflow (.github/workflows/main.yml) that automates the following:

Continuous Integration (CI): On every push to the main branch, the workflow installs dependencies and runs flake8 to check for code quality and style issues.

Continuous Deployment (CD): The workflow is set up to deploy the application to Streamlit Community Cloud. You will need to configure the STREAMLIT_API_TOKEN in your repository's secrets for this to work.

