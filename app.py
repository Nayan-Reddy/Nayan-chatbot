import streamlit as st
import json
import os
import re
import joblib
import numpy as np
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gspread
import uuid
from datetime import datetime
import time
import spacy
import streamlit.components.v1 as components

# ----------------- DATABASE SETUP (Google Sheets Version) -----------------
@st.cache_resource
def init_db():
    """Initializes the connection to the Google Sheet."""
    # Authenticate using Streamlit's secrets, which you set up in secrets.toml
    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    
    # Open the sheet by its exact name
    spreadsheet = gc.open("Chatbot Logs") # <-- Make sure this name matches your Google Sheet's name
    
    # Select the first worksheet
    worksheet = spreadsheet.sheet1
    return worksheet

# Initialize the connection
worksheet = init_db()

def log_interaction(session_id, query, response, source, response_time_ms):
    """Appends a single interaction as a new row in the Google Sheet."""
    try:
        # Format the timestamp into a string to avoid timezone issues
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get the current number of rows to create a simple ID
        num_rows = len(worksheet.get_all_values())
        
        new_row = [
            num_rows,  # Simple ID, starting from 0
            session_id,
            timestamp_str,
            query,
            response,
            source,
            response_time_ms
        ]
        
        # Append the new row to the sheet
        worksheet.append_row(new_row, value_input_option='USER_ENTERED')
    except Exception as e:
        st.error(f"Google Sheets logging failed: {e}")


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Nayan’s AI Chatbot", layout="centered")

# Load environment variable (GitHub Token)
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

# OpenAI client (GPT-4o via GitHub)
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=token,
)
chat_model = "openai/gpt-4o"

# ----------------- LOAD DATA -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "fallback_qna.json"), "r", encoding="utf-8") as f:
    fallback_data = json.load(f)

fallback_embeddings = joblib.load(os.path.join(BASE_DIR, "fallback_embeddings.pkl"))

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-small-en", device="cpu")

embedder = load_embedder()

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy NER model."""
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# Define the valid parts of your name at a broader scope
VALID_NAME_PARTS = {"nayan", "reddy", "soma"}

def contains_other_person_name(text, nlp_model):
    """
    Checks if the text contains a person's name that isn't related to Nayan Reddy Soma.
    """
    doc = nlp_model(text.lower())
    for ent in doc.ents:
        # Check if the entity is a person
        if ent.label_ == "PERSON":
            entity_text = ent.text
            # Check if the found name contains any part of your full name
            is_relevant_person = any(part in entity_text for part in VALID_NAME_PARTS)
            
            # If the person is not relevant, it's someone else
            if not is_relevant_person:
                return True
    return False


# ----------------- CLEAN + FILTER -----------------
def clean(text):
    return re.sub(r"[^\w\s]", "", text.lower().replace("’", "'").strip())

SENSITIVE_KEYWORDS = set([
    "gay", "sexuality", "husband", "wife", "sex", "sex life","children",
    "boyfriend", "mental", "asshole", "bitch", "chutiya", "motherfucker", "mf",
    "religion", "caste", "photo", "picture", "handsome", "ugly", "appearance",
    "politics", "weekend", "saturday", "sunday", "vacation", "gym", "body", "six pack",
    "weight", "shirtless", "personal", "intimate"
])

def is_sensitive(user_input):
    cleaned_input = clean(user_input)
    return any(keyword in cleaned_input for keyword in SENSITIVE_KEYWORDS)

def sensitive_reply(user_input):
    cleaned = clean(user_input)
    if "gender" in cleaned:
        return "Nayan is male."
    elif "married" in cleaned or "wife" in cleaned or "husband" in cleaned or "children" in cleaned:
        return "Nayan is unmarried and has no children."
    elif "photo" in cleaned or "picture" in cleaned:
        return "There’s no photo available here, but I’d be happy to walk you through his professional journey."
    else:
        return "Let’s stay focused on Nayan’s professional background. Feel free to ask anything about his work, projects, or skills!"

# ----------------- FALLBACK MATCH -----------------
def get_best_fallback(user_input, nlp_model):
    user_clean = clean(user_input)
    if contains_other_person_name(user_input, nlp_model):
        return None
    user_vector = embedder.encode([user_clean], convert_to_tensor=True)[0]

    best_sim = 0
    best_answer = None
    matched_question = None
    for q, vec, ans in fallback_embeddings:
        sim = cosine_similarity([user_vector.cpu().numpy()], [vec])[0][0]
        if sim > best_sim:
            best_sim, best_answer, matched_question = sim, ans, q

    if best_sim >= 0.87: 
        user_words = set(user_clean.split())
        match_words = set(clean(matched_question).split())
        common_words = user_words.intersection(match_words)

        if len(common_words) > 0:
            return best_answer

    best_fuzzy_score = 0
    fuzzy_answer = None
    for item in fallback_data:
        for q in item["questions"]:
            score = fuzz.token_sort_ratio(user_clean, clean(q))
            if user_clean in clean(q) or clean(q) in user_clean: score += 15
            if score > best_fuzzy_score:
                best_fuzzy_score, fuzzy_answer = score, item["answer"]
    
    if best_fuzzy_score >= 90: return fuzzy_answer
    return None

# ----------------- UI HEADER -----------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }

        h1 {
            color: #34d058;
            text-align: center;
            margin-bottom: 0;
        }

        .subheader {
            text-align: center;
            color: #c9d1d9;
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 30px;
        }

        .chat-bubble {
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 16px;
            border-radius: 12px;
            margin: 12px 0;
            line-height: 1.6;
        }

        .user-bubble {
            background-color: #10b981;
            color: black;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 20px 0 10px auto;
            max-width: 80%;
            text-align: right;
        }

        .bot-bubble {
            background-color: rgba(0,0,0,0);
            color: #ffffff;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 10px 0 30px 0;
            max-width: 80%;
            text-align: left;
        }

        .stButton button {
            background-color: #10b981;
            color: black;
            border-radius: 24px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
        }

        hr {
            border: none;
            border-top: 1px solid #30363d;
            margin: 25px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='color: #10b981;'>🤖 Hi, I'm Nayan’s AI Assistant</h1>", unsafe_allow_html=True)
st.markdown('<div class="subheader">Ask me anything about Nayan and his work!</div>', unsafe_allow_html=True)


# ----------------- SESSION -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system", 
        "content": (
            "You are Nayan’s AI Assistant. You represent Nayan Reddy Soma — a data enthusiast, analyst, and project-driven learner. "
            "Answer every question on his behalf in a professional, friendly, and recruiter-friendly tone.\n\n"

            "👤 **Personal Information**:\n"
            "• Full Name: Nayan Reddy Soma\n"
            "• DOB: 19 May 2002 (Dynamically calculate age)\n"
            "• Gender: Male\n"
            "• Native: Adilabad, Telangana\n"
            "• Father's Name: Lacha Reddy (English Teacher)\n"
            "• Mother's Name: Pavani Reddy (Government Employee)\n"
            "• Marital Status: Unmarried\n"
            "• Siblings: None\n"
            "• Extrovert personality\n"
            "• Enjoys gaming, exploring new places, playing volleyball and badminton\n"
            "• Regional-level volleyball player\n"
            "• Does not smoke or drink\n"
            "• Likes clubbing occasionally\n"

            "🎓 **Education**:\n"
            "• 10th: Vikas Concept School, Hyderabad – 10 CGPA\n"
            "• 12th: Narayana Junior College, Hyderabad – 88.4%\n"
            "• B.Tech: St. Martin’s Engineering College – 7.83 CGPA\n"
            "• Branch: Computer Science (AI & ML specialization)\n"
            "• Rejected full-stack development offer from ExcelR after graduation to prepare for CAT\n"
            "• Scored 96.42 percentile in CAT (General category, not selected for IIM)\n"
            "• Discovered passion for business insights, communication, stakeholder analysis — hence moved to data analytics\n"

            "💼 **Work Experience**:\n"
            "• No formal job experience, but has done multiple real-world analytics projects\n"
            "• Comfortable with working full-time, no notice period, and can join immediately\n"
            "• Open to relocation and short-term international work, but prefers to stay in India long term\n"
            "• No other job offers currently — actively looking for a good team and opportunity to grow\n"
            "• Doesn’t focus only on salary; wants learning and culture fit\n"

            "🛠️ **Skills & Tools**:\n"
            "• Power BI (ETL, DAX, RLS, Bookmarks, KPIs, Drill-through, Tooltips, Parameters)\n"
            "• SQL (Joins, Subqueries, CTEs, Window Functions, Optimization)\n"
            "• Python (Pandas, Matplotlib, NumPy)\n"
            "• Excel (PivotTables, VLOOKUP, INDEX-MATCH, Data Cleaning, Conditional Formatting)\n"
            "• FastAPI, MySQL\n"

            "📊 **Projects**:\n"

            "1. **Business 360 Power BI Dashboard**:\n"
            "- Analyzed 1.8M+ rows of data from Sales, Marketing, Finance, Supply Chain, and Executive views\n"
            "- Built with snowflake schema and custom DAX measures (YoY, moving averages, forecast errors)\n"
            "- Key Features: Dynamic slicers, maps, drilldowns, advanced KPIs, custom tooltips\n"
            "- Created for CXOs to make informed decisions\n"
            "- Technologies: Power BI, Power Query, DAX\n"

            "2. **Expense Tracker Management App**:\n"
            "- Tech Stack: FastAPI + Streamlit + MySQL\n"
            "- Session-based personalized expense logging and analytics\n"
            "- Visual dashboards show monthly trends, category-wise spending, budget utilization\n"
            "- Switches between demo mode and per-user analytics dynamically\n"
            "- Used: Pandas, Matplotlib, MySQL queries, Streamlit frontend\n"

            "3. **Sales Tracker in Excel**:\n"
            "- Built an Excel dashboard for sales, product, and regional performance\n"
            "- Used advanced formulas, conditional formatting, and slicers\n"
            "- Focused on automation, dynamic filtering, and clear visual communication\n"

            "4. **SQL Ad-hoc Business Analysis**:\n"
            "- Used MySQL to analyze a 1.4M+ row transactional database\n"
            "- Delivered insights for 10+ ad-hoc business questions for executive stakeholders\n"
            "- Example Insights:\n"
            "   • Product offerings grew by 36.33% in 2021 vs 2020\n"
            "   • Retailers contributed 73.21% of total revenue\n"
            "   • Sample SQL queries include joins, window functions, group by, ranking, sales drop detection\n"

            "🎯 **Interview-readiness**:\n"
            "• Personalized answers for 65+ behavioral and background questions are preloaded (e.g., 'Why should we hire you?')\n"
            "• The assistant should match similar variants like: 'Do you have work experience?' → 'Why no work experience?'\n"
            "• The assistant is trained to smartly fallback to exact answers when relevant or use OpenAI for everything else\n"
            "• If recruiter asks 'show me your projects' — the assistant must include project links:\n"
            "   • Power BI Live Dashboard: [https://app.powerbi.com/view?r=eyJrIjoiYzFkZGY3NGUtMWIwYy00YjZmLWIzMDYtYjQyMjkxNGRhN2NmIiwidCI6ImM2ZTU0OWIzLTVmNDUtNDAzMi1hYWU5LWQ0MjQ0ZGM1YjJjNCJ9]\n"
            "   • Expense Tracker App: [https://expense-tracker-frontend-nayan-reddy.streamlit.app/]\n"
            "   • Excel Demo File: [https://1drv.ms/x/c/e4ca29151a0a4ec4/EeJ0-_SJhOZIjam0emzh_ccBfDKFeWhL2IMsVI7DXtXB0Q?e=jisfwq]\n"

            "🧠 **Behavioral Summary**:\n"
            "• Strong communication, team-oriented, and open to feedback\n"
            "• Loves learning from peers and collaborating on real data problems\n"
            "• Passionate about making data useful to decision-makers\n"

            "🛡️ **Sensitive Topic Policy**:\n"
            "If users ask questions about:\n"
            "- Gender, sexuality, relationship status, sex life, weekend plans, marriage, children, or any question that feels overly personal, informal, or inappropriate\n"
            "- Caste, religion, appearance, photo, or political views\n"
            "- Inappropriate, sarcastic, or offensive phrasing\n"
            "\n"
            "Then respond briefly and professionally. Do not provide full introductions or irrelevant answers. Instead:\n"
            "• 'Nayan is male.'\n"
            "• 'He is unmarried and has no children.'\n"
            "• 'Let’s stay focused on Nayan’s work and projects.'\n"
            "• 'That’s a bit personal — happy to answer anything about his professional background.'\n"
            "\n"
            "Avoid judgmental or speculative responses. Stay friendly, respectful, and focused on the recruiter’s intent.\n"

            "Whenever possible, answer proactively by highlighting Nayan’s achievements, projects, and personality.\n"
            "If users ask vague things like 'tools used', 'project details', or just 'Power BI', intelligently respond using project context.\n"
            "Never say 'I don’t know'. Always be confident, helpful, and insightful."
        )
    }]

    st.session_state.show_prompts = True
if "fallback_history" not in st.session_state:
    st.session_state.fallback_history = []
if "last_fallback_qna" not in st.session_state:
    st.session_state.last_fallback_qna = None

# --- CONFIGURABLE BRANDING ---
# Replace these with your own details
logo_url = "https://raw.githubusercontent.com/Nayan-Reddy/Nayan-Resume/refs/heads/main/n.png"
link_url = "https://nayan-reddy.github.io/Nayan-Resume/"
brand_text = "Created by Nayan Reddy"


# --- HTML & CSS & JS for the component ---
branding_html = f"""
<style>
    /* All the CSS you provided */
    .branding {{
        position: fixed;
        bottom: 10px;
        right: 16px;
        z-index: 1000;
        cursor: pointer;
    }}
    .branding a {{
        text-decoration: none;
    }}
    .branding-container {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        background-color: #f0f2f6;
        border-radius: 25px; /* Pill shape */
        height: 50px;
        width: 50px;
        overflow: hidden;
        transition: width 0.4s ease-in-out;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    .brand-text {{
        font-size: 14px;
        font-weight: 500;
        color: #4f4f4f;
        white-space: nowrap;
        opacity: 0;
        padding-right: 12px;
        transition: opacity 0.3s ease-in-out 0.1s;
    }}
    .brand-logo {{
        width: 50px;
        height: 50px;
        object-fit: cover;
        flex-shrink: 0;
        border-radius: 50%;
    }}
    .branding.expanded .branding-container {{
        width: 200px; /* Adjust as needed for your text */
    }}
    .branding.expanded .brand-text {{
        opacity: 1;
    }}
</style>

<div class="branding" id="branding">
    <a href="{link_url}" target="_blank">
        <div class="branding-container">
            <span class="brand-text">{brand_text}</span>
            <img src="{logo_url}" class="brand-logo" alt="Logo">
        </div>
    </a>
</div>

<script>
    const branding = document.getElementById('branding');
    let hideTimeout;

    function showBranding() {{
        clearTimeout(hideTimeout);
        branding.classList.add('expanded');
        hideTimeout = setTimeout(() => {{
            branding.classList.remove('expanded');
        }}, 4000);
    }}

    // Initial animation on page load
    setTimeout(showBranding, 1500);

    // Re-trigger animation on hover or click
    branding.addEventListener('mouseenter', showBranding);
    branding.addEventListener('click', showBranding);
</script>
"""

# --- Render the component in Streamlit ---
components.html(branding_html, height=70)

user_input = st.chat_input("Ask a question...")


if user_input and st.session_state.show_prompts:
    st.session_state.show_prompts = False

if st.session_state.show_prompts:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 25px; color: #c9d1d9; font-size: 15px;'>
        💡 Try asking: <br>
        “Can you introduce yourself?”<br>
        “What are your interests or hobbies outside of work?”<br>
        “Tell me about your projects or the tools you’ve worked with?”
    </div>
    """, unsafe_allow_html=True)

# ----------------- FOLLOW-UP CHECK -----------------
def is_follow_up(user_input):
    followup_keywords = ["more", "elaborate", "explain", "details", "detail", "expand",
        "why", "how", "what else", "what was", "what did", "tell me more", "about it", "that one",
        "which one", "who", "it", "that", "this", "he", "she", "they", "them",
        "tell me more", "more info", "how long", "who was involved", "what was the tool",
        "what’s the tech", "what technology", "did it work"]
    return any(kw in clean(user_input) for kw in followup_keywords)

def ask_gpt_with_context(user_input, fallback_context=None):
    messages = st.session_state.messages[:1]
    if fallback_context:
        for q, a in fallback_context:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Could not fetch response.\n\n**Error:** {e}"

# ----------------- PROCESS QUESTION -----------------
if user_input:
    st.session_state.show_prompts = False
    st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.messages.append({"role": "user", "content": user_input})
    # --- Start Logging Additions ---
    start_time = time.time()
    reply = ""
    response_source = ""
    # --- End Logging Additions ---
    if is_sensitive(user_input):
        reply = sensitive_reply(user_input)
        response_source = "sensitive_filter"
    elif is_follow_up(user_input) and st.session_state.last_fallback_qna:
        last_q, last_a = st.session_state.last_fallback_qna
        context = [
            {"role": "user", "content": last_q},
            {"role": "assistant", "content": last_a},
            {"role": "user", "content": user_input}
        ]
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=[st.session_state.messages[0]] + context
                )
                reply = response.choices[0].message.content
                response_source = "llm_follow_up"
            except Exception as e:
                reply = f"❌ Could not fetch response.\n\n**Error:** {e}"
                response_source = "error"
    else:
        fallback = get_best_fallback(user_input, nlp)
        if fallback:
            reply = fallback
            response_source = "fallback"
            st.session_state.fallback_history.append((user_input, reply))
            st.session_state.fallback_history = st.session_state.fallback_history[-5:]
            st.session_state.last_fallback_qna = (user_input, fallback)
        else:
            with st.spinner("Thinking..."):
                try:
                    fallback_qna_context = []
                    for q, a in st.session_state.fallback_history[-5:]:
                        fallback_qna_context.append({"role": "user", "content": q})
                        fallback_qna_context.append({"role": "assistant", "content": a})
                    all_context = [st.session_state.messages[0]] + fallback_qna_context + st.session_state.messages[-5:]
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=all_context
                    )
                    reply = response.choices[0].message.content
                    response_source = "llm_general"
                except Exception as e:
                    reply = f"❌ Could not fetch response.\n\n**Error:** {e}"
                    response_source = "error"
    # --- Final Logging Step ---
    end_time = time.time()
    response_time_ms = int((end_time - start_time) * 1000)
    log_interaction(
        session_id=st.session_state.session_id,
        query=user_input,
        response=reply,
        source=response_source,
        response_time_ms=response_time_ms
    )
    # --- End Logging ---
    st.session_state.messages.append({"role": "assistant", "content": reply})
    max_pairs = 5
    system_message = st.session_state.messages[0]
    chat_history = st.session_state.messages[1:]
    trimmed = chat_history[-(max_pairs * 2):]
    st.session_state.messages = [system_message] + trimmed

    # Stylish divider
    st.markdown("""
        <div style="margin: 20px auto; width: 100px; height: 2px; background: linear-gradient(to right, #10b981, #1f2937); border-radius: 2px;"></div>
        """, unsafe_allow_html=True)

    # Show assistant reply
    st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)

