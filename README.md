# AI-Powered Chatbot 🤖
<div align="center">

*An intelligent, self-aware AI chatbot that serves as a dynamic, interactive portfolio for a user, powered by a sophisticated RAG pipeline and advanced NLP.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-74AA9C?style=for-the-badge&logo=openai)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge&logo=spacy)
![Google Sheets](https://img.shields.io/badge/Google_Sheets-Logging-34A853?style=for-the-badge&logo=google-sheets)
![Plotly](https://img.shields.io/badge/Plotly-Visuals-blueviolet?style=for-the-badge&logo=plotly)

</div>

---

## 📈 Live Demos

<div align="center">
  <strong>Experience the chatbot in action: <a href="https://nayan-chatbot.streamlit.app/">➡️ Interact with the Live Demo Here</a></strong>
</div>

<p align="center">
  <img src="[LINK TO YOUR CHATBOT GIF]" alt="Nayan's AI Assistant Demo GIF" width="800"/>
</p>

<div align="center">
  <strong>View the live analytics of the chatbot: <a href="https://nayan-chatbot-analytics.streamlit.app/">➡️ Live Analytics Dashboard</a></strong>
</div>
  
<p align="center">
  <img src="[LINK TO YOUR ANALYTICS DASHBOARD GIF]" alt="Analytics Dashboard Demo GIF" width="800"/>
</p>

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Core Features](#-core-features)
- [Architecture & Tech Stack](#-tech-stack)
- [Setup and Local Installation](#-getting-started)
- [Environment Configuration](#-environment-configuration)
- [Challenges & Learnings](#-challenges--learnings)
- [Future Enhancements](#-future-enhancements)
- [Contact](#-contact)

---

## 🌟 Introduction

**Nayan's AI Assistant** is a full-stack chatbot application designed to be more than just an information source—it's an intelligent, interactive representation of my professional profile. It leverages a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline to provide recruiters and collaboraters an accurate, context-aware answers about my skills, projects, and background.

This project was built to demonstrate a deep understanding of modern AI application development, from sophisticated NLP-powered guardrails and conversational memory to a complete, cloud-based analytics pipeline for monitoring user interactions in real-time.

---

## ✨ Core Features

This project is more than a simple Q&A bot. It's an end-to-end showcase of modern AI application development.

* **🧠 Hybrid RAG System:** A multi-step retrieval strategy ensures fast, accurate, and relevant answers. The logic prioritizes responses in the following order:
    1.  **High-Confidence Semantic Match:** Uses a `BAAI/bge-small-en` sentence transformer to find the most similar question from a pre-computed vector database. An answer is returned if the cosine similarity score is **≥ 0.87**.
    2.  **Lexical Fuzzy Match:** If semantic search fails, it uses `fuzzywuzzy`'s token sort ratio to find a close match. An answer is returned if the score is **≥ 90**.
    3.  **Generative Fallback with Context:** For novel or nuanced questions, the bot uses **OpenAI's GPT-4o** model, providing it with the recent conversation history for context.

* **🛡️ Intelligent Guardrails:**
    * **NER-Powered Scope Control:** Utilizes `spaCy` for Named Entity Recognition (NER) to detect if a question mentions another person's name. This prevents the bot from answering questions that are outside its scope of representing Nayan Reddy Soma.
    * **Sensitive Topic Filtering:** A custom keyword filter deflects inappropriate or overly personal questions with professional, pre-defined responses.

* **📊 Real-time Analytics Pipeline:**
    * Every user interaction with the live chatbot is logged in real-time to a **Google Sheet** using the `gspread` API.
    * Data points captured include `session_id`, `timestamp`, `user_query`, `final_response`, `response_source` (e.g., fallback, llm_general), and `response_time_ms`.

* **📈 Decoupled Analytics Dashboard:**
    * A separate Streamlit app (`analytics.py`) reads the live data from Google Sheets to provide insights on:
        * **KPIs:** Total users, total questions, average questions per user, and average response time.
        * **Performance:** A pie chart showing the distribution of response sources (how often the RAG system vs. the LLM provides an answer).
        * **Engagement:** A bar chart of daily usage and a table of the most frequently asked questions.

* **🗣️ Context-Aware Follow-ups:** The chatbot remembers the context of the last interaction, allowing it to handle follow-up questions like "tell me more about that" or "why was that important?" with high relevance.

---

## 🛠️ Architecture & Tech Stack

This project is built with a modern, end-to-end Python stack designed for performance and scalability.

| Category                | Technology / Library                                                                        | Purpose                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Web Framework** | **Streamlit** | For building the interactive chat UI and the analytics dashboard.                      |
| **Backend Logic** | **Python 3.10+** | Core application logic, data processing, and integrations.                             |
| **NLP (Retrieval)** | **`sentence-transformers`** | To generate vector embeddings for semantic search (`BAAI/bge-small-en` model).         |
|                         | **`scikit-learn`** | For calculating cosine similarity between text embeddings.                             |
|                         | **`fuzzywuzzy`** | For lexical-based fuzzy string matching as a secondary retrieval layer.                |
| **NLP (Guardrails)** | **`spaCy` (`en_core_web_sm`)** | For Named Entity Recognition (NER) to power the smart scope-control guardrail.         |
| **Generative AI** | **OpenAI GPT-4o** | The final generative layer for handling novel and conversational questions.            |
| **Database & Logging** | **Google Sheets API (`gspread`)** | A robust and free solution for real-time logging and data collection from the cloud.   |
| **Data Analysis** | **`pandas`, `plotly`** | For data manipulation and creating visualizations in the analytics dashboard.          |
| **Deployment** | **Streamlit Community Cloud** | For hosting the live chatbot and analytics dashboard.                                  |
| **Dependencies** | **`joblib`, `numpy`** | For serializing/deserializing the embedding file and numerical operations.             |
| **Environment Mgmt** | **`python-dotenv`** | To manage local environment variables.                                                 |

---

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/Nayan-Reddy/Nayan-chatbot.git)
    cd your-repo-name
    ```

2.  **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The `requirements.txt` file is configured to install CPU-specific versions of PyTorch and the direct spaCy model for efficiency.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Embeddings:**
    This is a crucial one-time preprocessing step. Run this script to create the `fallback_embeddings.pkl` file from your Q&A data.
    ```bash
    python generate_embeddings.py
    ```

5.  **Configure Environment Variables/Secrets:**
    - Follow the instructions in the [Environment Configuration](#-environment-configuration) section below to set up your API keys and credentials.

6.  **Run the Apps:**
    You can now run the chatbot and the analytics dashboard locally.

    * **Run the Chatbot:**
        ```bash
        streamlit run app.py
        ```
    * **Run the Analytics Dashboard:**
        ```bash
        streamlit run analytics.py
        ```

---

## 🔑 Environment Configuration

The application requires two separate files for credentials:

1.  **For the GitHub API Token (via OpenAI):**
    - Create a file named `.env` in the project's root directory.
    - Add your GitHub token (used for the OpenAI proxy):
      ```ini
      # .env
      GITHUB_TOKEN="ghp_YOUR_TOKEN_HERE"
      ```

2.  **For Google Sheets Logging:**
    - Create a folder named `.streamlit` in the project's root directory.
    - Inside that folder, create a file named `secrets.toml`.
    - Paste your Google Cloud Platform service account JSON credentials here. This is used by both `app.py` and `analytics.py`.
      ```toml
              # .streamlit/secrets.toml

        [gcp_service_account]
        type = "service_account"
        project_id = "your-gcp-project-id"
        private_key_id = "your-private-key-id"
        private_key = "-----BEGIN PRIVATE KEY-----\nYOUR-PRIVATE-KEY\n-----END PRIVATE KEY-----\n"
        client_email = "your-client-email@your-gcp-project-id.iam.gserviceaccount.com"
        client_id = "your-client-id"
        # ... and so on for the rest of the JSON key file.
        ```


---

## 🧠 Challenges & Learnings

This project involved solving several key challenges, leading to significant learnings:

1.  **Challenge:** The initial bot would incorrectly answer questions about other people (e.g., "What are Akash's projects?").
    - **Learning:** I evolved the solution from a simple, brittle keyword list to a sophisticated **NER-based guardrail using `spaCy`**. This taught me the power of using robust NLP models to create intelligent, context-aware application rules.

2.  **Challenge:** How to monitor and log interactions on a deployed, serverless platform like Streamlit Cloud where local files are ephemeral.
    - **Learning:** I designed and implemented a **real-time data pipeline** using the Google Sheets API. This involved understanding service account authentication, secure secrets management, and building a decoupled system where the live app acts as a data producer and a separate analytics app as a consumer.

3.  **Challenge:** The bot's conversational memory was initially poor, relying on complex and unreliable logic to handle follow-ups.
    - **Learning:** I refactored the state management by implementing a **unified chat history**. This simplified the codebase immensely and leveraged the inherent context-handling capabilities of the LLM, resulting in a much smarter and more natural conversational flow.

---



## 🔮 Future Improvements

This project has a strong foundation that can be extended with even more features:

* **User Feedback System:** Add thumbs-up/down buttons to log user satisfaction with responses directly into the Google Sheet for finer-grained analysis.
* **Advanced Analytics:** Use semantic clustering on the logged questions to identify common user intents that are not yet covered in the `fallback_qna.json`.
* **Multi-Modal Capabilities:** Integrate tools to display images of projects or architecture diagrams directly in the chat when asked.

---

## 📫 Get In Touch

I'm a passionate data enthusiast actively seeking opportunities in data analytics and AI. If you're impressed by this project or have any questions, I'd love to connect!

* **Email:** [Your Email Address]
