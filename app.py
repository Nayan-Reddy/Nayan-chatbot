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
import speech_recognition as sr
import io
import threading
from pydub import AudioSegment
import webrtcvad
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase


# ----------------- VAD & AUDIO CONSTANTS -----------------
VAD_AGGRESSIVENESS = 3  # How aggressive VAD is (0-3)
SAMPLE_RATE = 16000     # 16k Hz is required by SpeechRecognition
FRAME_DURATION = 30     # ms
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# ----------------- REAL-TIME AUDIO PROCESSOR CLASS -----------------
class VADAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.lock = threading.Lock()
        self.speech_buffer = io.BytesIO()
        self.is_speaking = False
        self.silence_frames = 0

    def _resample_and_convert(self, audio_chunk: AudioSegment) -> bytes:
        """Ensure audio is 16k Hz mono for VAD."""
        audio_chunk = audio_chunk.set_frame_rate(SAMPLE_RATE).set_channels(1)
        return audio_chunk.raw

    def recv(self, frame):
        audio_segment = AudioSegment(
            data=frame.to_ndarray(format="s16").tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )
        
        processed_audio = self._resample_and_convert(audio_segment)
        
        with self.lock:
            for i in range(0, len(processed_audio), CHUNK_SIZE * 2):
                chunk = processed_audio[i:i + CHUNK_SIZE * 2]
                if len(chunk) < CHUNK_SIZE * 2:
                    break
                
                is_speech = self.vad.is_speech(chunk, SAMPLE_RATE)
                
                if self.is_speaking:
                    self.speech_buffer.write(chunk)
                    if not is_speech:
                        self.silence_frames += 1
                        if self.silence_frames * FRAME_DURATION > 1500:
                            self.is_speaking = False
                            self.finalize_speech()
                    else:
                        self.silence_frames = 0
                elif is_speech:
                    self.is_speaking = True
                    self.silence_frames = 0
                    self.speech_buffer.write(chunk)
        
        return frame

    def finalize_speech(self):
        """Called when speech ends. Transcribes the buffered audio."""
        recognizer = sr.Recognizer()
        self.speech_buffer.seek(0)
        
        if self.speech_buffer.getbuffer().nbytes == 0:
            st.session_state.transcribed_text = ""
            return

        audio_data = sr.AudioData(
            frame_data=self.speech_buffer.read(),
            sample_rate=SAMPLE_RATE,
            sample_width=2 # 16-bit
        )
        
        try:
            text = recognizer.recognize_google(audio_data, language="en-IN")
            st.session_state.transcribed_text = text
        except (sr.UnknownValueError, sr.RequestError):
            st.session_state.transcribed_text = ""
        
        self.speech_buffer = io.BytesIO()


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
def get_best_fallback(user_input):
    user_clean = clean(user_input)
    user_vector = embedder.encode([user_clean], convert_to_tensor=True)[0]

    best_sim = 0
    best_answer = None

    for q, vec, ans in fallback_embeddings:
        sim = cosine_similarity([user_vector.cpu().numpy()], [vec])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_answer = ans

    if best_sim >= 0.82:
        return best_answer

    best_fuzzy_score = 0
    fuzzy_answer = None
    for item in fallback_data:
        for q in item["questions"]:
            score = fuzz.token_sort_ratio(user_clean, clean(q))
            if user_clean in clean(q) or clean(q) in user_clean:
                score += 15
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                fuzzy_answer = item["answer"]

    if best_fuzzy_score >= 90:
        return fuzzy_answer

    return None

# ----------------- VOICE INPUT FUNCTION -----------------


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

if "show_prompts" not in st.session_state:
    st.session_state.show_prompts = True
if "fallback_history" not in st.session_state:
    st.session_state.fallback_history = []
if "last_fallback_qna" not in st.session_state:
    st.session_state.last_fallback_qna = None
# In your SESSION state block
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None

# Initialize user_input variable
user_input = None

# --- NEW: REAL-TIME VAD INPUT UI ---
with st._bottom:
    if st.session_state.is_listening:
        # --- LISTENING STATE ---
        st.info("🎤 Listening... I'll stop automatically when you pause.")
        
        webrtc_streamer(
            key="vad_recorder",
            mode=WebRtcMode.SEND_ONLY,
            audio_processor_factory=VADAudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
        )
        
    else:
        # --- DEFAULT TEXT INPUT STATE ---
        cols = st.columns([0.9, 0.1])
        with cols[0]:
            user_input_text = st.chat_input("Ask a question...", key="chat_input")
            if user_input_text:
                user_input = user_input_text
        with cols[1]:
            if st.button("🎤", help="Speak your question", use_container_width=True):
                st.session_state.is_listening = True
                st.rerun()

# Check if transcription has come back from the audio processor
if st.session_state.transcribed_text is not None:
    if st.session_state.transcribed_text: # Ensure it's not an empty string
        user_input = st.session_state.transcribed_text
    
    st.session_state.transcribed_text = None  # Reset for next use
    st.session_state.is_listening = False # Switch back to text input UI
    st.rerun()
                
                
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

    if is_sensitive(user_input):
        reply = sensitive_reply(user_input)
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
            except Exception as e:
                reply = f"❌ Could not fetch response.\n\n**Error:** {e}"
    else:
        fallback = get_best_fallback(user_input)
        if fallback:
            reply = fallback
            st.session_state.fallback_history.append((user_input, reply))
            st.session_state.fallback_history = st.session_state.fallback_history[-5:]
            st.session_state.last_fallback_qna = (user_input, fallback)
        else:
            with st.spinner("Thinking..."):
                try:
                    fallback_qna_context = [
                        {"role": "user", "content": q} if i % 2 == 0 else {"role": "assistant", "content": a}
                        for i, (q, a) in enumerate(st.session_state.fallback_history[-5:])
                        for _ in (0, 1)
                    ]
                    all_context = [st.session_state.messages[0]] + fallback_qna_context + st.session_state.messages[-5:]
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=all_context
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"❌ Could not fetch response.\n\n**Error:** {e}"

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

