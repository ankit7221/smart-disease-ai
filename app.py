import streamlit as st
import joblib
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load environment variables (for API keys) ---
load_dotenv()

# --- Configuration Paths ---
MODEL_DIR = 'model/'
NAIVE_BAYES_MODEL_PATH = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SYMPTOM_BINARIZER_PATH = os.path.join(MODEL_DIR, 'symptom_binarizer.pkl')
CSS_FILE_PATH = 'style.css' # Define path to your external CSS file

# --- Load ML Models and Binarizer ---
@st.cache_resource(show_spinner=False) # Manage spinners manually for custom messages
def load_ml_models():
    """Loads the pre-trained ML models and symptom binarizer."""
    #st.info("🧠 Loading essential models for accurate predictions...")
    try:
        nb_model = joblib.load(NAIVE_BAYES_MODEL_PATH)
        rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
        mlb = joblib.load(SYMPTOM_BINARIZER_PATH)
        return nb_model, rf_model, mlb
    except FileNotFoundError:
        st.error("🚨 Error: Model files or symptom binarizer not found. Please run 'python train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}. Please check your model directory and file integrity.")
        st.stop()

# --- Load Gemini model ---
@st.cache_resource(show_spinner=False) # Manage spinners manually for custom messages
def load_gemini_model():
    """Configures and loads the Google Gemini model for AI advice."""
    #st.info("✨ Connecting to powerful AI Advisor...")
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        print(f"Loaded Google Generative Model: {model.model_name}") # For terminal debug
        return model
    except Exception as e:
        st.error(f"❌ Error connecting to AI Advisor: {e}. Please check your GOOGLE_API_KEY and ensure the Gemini API is enabled for your project.")
        st.stop()


# --- Load models and Gemini ---
nb_model, rf_model, mlb = load_ml_models()
gemini_model = load_gemini_model()


# --- Initialize session state ---
if 'predicted_disease_for_gemini' not in st.session_state:
    st.session_state.predicted_disease_for_gemini = "a general health concern"
    st.session_state.predicted_confidence = 0.0
    st.session_state.predicted_model = "None"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initial_gemini_prompt_sent' not in st.session_state:
    st.session_state.initial_gemini_prompt_sent = False
if 'initial_ai_response_display' not in st.session_state:
    st.session_state.initial_ai_response_display = ""
if 'chat_input_value' not in st.session_state:
    st.session_state.chat_input_value = ""
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'follow_up_started' not in st.session_state:
    st.session_state.follow_up_started = False
# New session states for symptom details
if 'symptom_details' not in st.session_state:
    st.session_state.symptom_details = {} # Stores {symptom: {'severity': X, 'duration': Y}}


# --- Function to load CSS from file ---
def load_css(file_name):
    """Reads a CSS file and injects its content into the Streamlit app."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        # Always explicitly load Font Awesome if needed, as @import might be blocked
        st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)
        print(f"CSS file '{file_name}' loaded successfully.") # Debug print to terminal
    except FileNotFoundError:
        st.error(f"Error: CSS file '{file_name}' not found. Please ensure it's in the same directory as app.py.")
        print(f"ERROR: CSS file '{file_name}' not found!") # Debug print to terminal
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
        print(f"ERROR: Exception loading CSS: {e}") # Debug print to terminal

# --- Streamlit UI Configuration & Styling ---
st.set_page_config(
    page_title="Smart Disease & Health AI",
    layout="wide",
)

# Load and inject custom CSS from style.css
load_css(CSS_FILE_PATH)


# --- Main Application Layout ---
st.markdown('<h1 class="main-header"><i class="fas fa-microscope"></i> Smart Disease & Health AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Get instant predictions based on your symptoms and receive AI-powered general health advice. Your personal health companion.</p>', unsafe_allow_html=True)

# Use columns for main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Input Symptoms")
    st.markdown("Select all the symptoms you are currently experiencing:")

    all_symptoms = sorted(mlb.classes_)
    selected_symptoms = st.multiselect(
        "Symptoms:",
        all_symptoms,
        key="symptom_selector",
        placeholder="Search or select symptoms...",
        help="Type to search for symptoms or select from the dropdown."
    )

    # --- New: Symptom Severity and Duration Inputs ---
    st.markdown("---")
    st.markdown("#### Details for Selected Symptoms:")
    symptom_data_for_ai = [] # To collect details for the AI prompt

    # Ensure st.session_state.symptom_details is updated only for currently selected symptoms
    # and cleared for unselected ones.
    current_details = {}
    for symptom in selected_symptoms:
        # Retain old values if available, otherwise set defaults
        current_details[symptom] = st.session_state.symptom_details.get(symptom, {'severity': 5, 'duration': 1})

        col_symptom, col_severity, col_duration = st.columns([2, 1, 1])
        with col_symptom:
            st.write(f"**{symptom.replace('_', ' ').title()}**")
        with col_severity:
            current_details[symptom]['severity'] = st.slider(
                f"Severity ({symptom}):",
                1, 10,
                value=current_details[symptom]['severity'],
                key=f"severity_{symptom}",
                help="Rate the severity from 1 (mild) to 10 (severe)."
            )
        with col_duration:
            current_details[symptom]['duration'] = st.number_input(
                f"Duration ({symptom}) (days):",
                min_value=0,
                max_value=365,
                value=current_details[symptom]['duration'],
                key=f"duration_{symptom}",
                help="How many days have you experienced this symptom?",
                step=1
            )
        symptom_data_for_ai.append(
            f"{symptom.replace('_', ' ').title()} (Severity: {current_details[symptom]['severity']}/10, Duration: {current_details[symptom]['duration']} days)"
        )
    st.session_state.symptom_details = current_details # Update session state

    # --- End New Inputs ---


    if st.button("Predict Disease ▶️", key="predict_button_main"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom to get a prediction.")
        else:
            input_symptoms_list = [selected_symptoms]
            input_vector = mlb.transform(input_symptoms_list)

            nb_prediction = nb_model.predict(input_vector)[0]
            nb_proba = nb_model.predict_proba(input_vector)[0]
            nb_confidence = max(nb_proba) * 100

            rf_prediction = rf_model.predict(input_vector)[0]
            rf_proba = rf_model.predict_proba(input_vector)[0]
            rf_confidence = max(rf_proba) * 100

            if nb_confidence >= rf_confidence:
                st.session_state.predicted_disease_for_gemini = nb_prediction.replace('_', ' ').title()
                st.session_state.predicted_confidence = nb_confidence
                st.session_state.predicted_model = "Naive Bayes"
            else:
                st.session_state.predicted_disease_for_gemini = rf_prediction.replace('_', ' ').title()
                st.session_state.predicted_confidence = rf_confidence
                st.session_state.predicted_model = "Random Forest"

            st.session_state.prediction_made = True
            st.session_state.chat_history = [] # Clear chat history
            st.session_state.initial_gemini_prompt_sent = False # Reset initial prompt status
            st.session_state.initial_ai_response_display = "" # Clear previous initial AI response
            st.session_state.chat_input_value = ""
            st.session_state.follow_up_started = False # Reset follow-up status
            # Store symptoms and their details for the AI to use in the initial prompt
            st.session_state.symptoms_for_gemini_prompt = symptom_data_for_ai


            st.rerun()

with col2:
    if not st.session_state.prediction_made: # If no prediction yet
        st.subheader("2. Prediction Results")
        st.info("💡 Select symptoms on the left and click 'Predict Disease' to see results here.")

        st.subheader("3. AI Health Advisor Chat")
        st.info("💡 First, get a disease prediction to enable the AI Advisor.")
    else: # Prediction has been made
        st.subheader("2. Prediction Results")
        # Always display both model predictions when a prediction is made
        input_vector_display = mlb.transform([selected_symptoms]) if selected_symptoms else [[]]
        nb_prediction_display = nb_model.predict(input_vector_display)[0].replace('_', ' ').title() if selected_symptoms else 'N/A'
        nb_confidence_display = max(nb_model.predict_proba(input_vector_display)[0]) * 100 if selected_symptoms else 0.0

        rf_prediction_display = rf_model.predict(input_vector_display)[0].replace('_', ' ').title() if selected_symptoms else 'N/A'
        rf_confidence_display = max(rf_model.predict_proba(input_vector_display)[0]) * 100 if selected_symptoms else 0.0

        st.markdown(f"""
            <div class="prediction-box">
                <h4><i class="fas fa-chart-bar"></i> AI Prediction 1:</h4>
                <p>Predicted Condition: <span>{nb_prediction_display}</span></p>
                <p>Confidence: {nb_confidence_display:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="prediction-box">
                <h4><i class="fas fa-project-diagram"></i> AI Prediction 2:</h4>
                <p>Predicted Condition: <span>{rf_prediction_display}</span></p>
                <p>Confidence: {rf_confidence_display:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        st.success(f"✅ Best prediction for AI advice: **{st.session_state.predicted_disease_for_gemini}** (Confidence: {st.session_state.predicted_confidence:.2f}% from {st.session_state.predicted_model} Model)")


        st.subheader("3. AI Health Advisor Chat")
        st.info(f"💡 The AI will now provide general health insights for: **{st.session_state.predicted_disease_for_gemini}**. Please remember this is general information, not medical advice.")

        # --- Initial AI Question and Response Flow ---
        if not st.session_state.initial_gemini_prompt_sent:
            initial_user_query = st.text_input(
                "Your Initial Question:",
                key="initial_gemini_query_input",
                placeholder=f"e.g., What are home remedies for {st.session_state.predicted_disease_for_gemini}? or Should I rest?",
                help="Type your first question about the predicted health concern. This will start the conversation."
            )

            if st.button("Get Initial AI Insights ▶️", key="get_initial_ai_insights_button"):
                if initial_user_query:
                    with st.spinner("✨ Generating your initial insights..."):
                        try:
                            symptom_details_str = ", ".join(st.session_state.symptoms_for_gemini_prompt) if st.session_state.symptoms_for_gemini_prompt else "no specific symptom details provided"

                            system_context_prompt = (
                                f"Given that the predicted general health concern (based on symptoms) is '{st.session_state.predicted_disease_for_gemini}'. "
                                f"The user has reported the following symptoms with their severity and duration: {symptom_details_str}. "
                                f"Please provide general health information or answer the following question: '{initial_user_query}'. "
                                "It is crucial that this information is general advice and is NOT a medical diagnosis, prescription, or substitute for professional medical advice. "
                                "Always advise consulting a healthcare professional for accurate diagnosis and treatment, especially considering the severity and duration of symptoms if applicable."
                            )

                            response_obj = gemini_model.generate_content(
                                contents=[{"role": "user", "parts": [{"text": system_context_prompt}]}],
                                generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=500)
                            )
                            response_text = response_obj.text

                            # Store the initial response for distinct display
                            st.session_state.initial_ai_response_display = response_text
                            st.session_state.initial_gemini_prompt_sent = True # Mark that initial response is displayed

                            # Store both user's initial query and AI's initial response in chat_history
                            # This ensures they are part of the context for follow-up questions
                            st.session_state.chat_history.append({"role": "user", "parts": [{"text": initial_user_query}]})
                            st.session_state.chat_history.append({"role": "model", "parts": [{"text": response_text}]})

                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error getting initial insights from AI Advisor: {e}")
                            st.warning("There might be an issue with the AI service or your request. Please try again or check your API key/project settings.")
                else:
                    st.warning("⚠️ Please type your initial question before clicking 'Get Initial AI Insights'.")
        else: # Initial AI response has been sent and displayed
            # Display the distinct initial AI response block
            st.markdown(f"""
                <div class="initial-ai-response-box">
                    <h4><i class="fas fa-robot"></i> AI Advisor Initial Insights:</h4>
                    <p>{st.session_state.initial_ai_response_display}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- Now, directly follow with the "Continue the Conversation" section and chat elements ---
            st.markdown("---")
            st.subheader("4. Continue the Conversation")

            # Only show the chat container if a follow-up has started
            if st.session_state.follow_up_started:
                chat_messages_html = "<div class='chat-container'>"
                # Display all messages in chat_history from the start, as it's now a continuous log
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        chat_messages_html += f"<div class='user-message'><b>You:</b> {message['parts'][0]['text']}</div>"
                    else:
                        chat_messages_html += f"<div class='ai-message'><b>AI Advisor:</b> {message['parts'][0]['text']}</div>"
                chat_messages_html += "</div>"
                st.markdown(chat_messages_html, unsafe_allow_html=True)
            else:
                 st.markdown("<p style='text-align: center; color: #888;'>Ask a follow-up question below to start a continuous conversation log.</p>", unsafe_allow_html=True)


            with st.form(key="chat_form"):
                chat_input = st.text_input(
                    "Your follow-up question:",
                    value=st.session_state.chat_input_value,
                    key="actual_chat_input_widget",
                    placeholder="Type your next question...",
                    help="Type your question to continue the conversation with the AI Advisor."
                )
                submit_button = st.form_submit_button("Send Message ▶️")

                if submit_button:
                    if chat_input:
                        with st.spinner("💬 AI Advisor is thinking..."):
                            try:
                                # Mark that follow-up has started
                                st.session_state.follow_up_started = True

                                # Append current chat input to chat history
                                st.session_state.chat_history.append({"role": "user", "parts": [{"text": chat_input}]})

                                # Send the entire history for context
                                response_obj = gemini_model.generate_content(
                                    contents=st.session_state.chat_history,
                                    generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=300)
                                )
                                response_text = response_obj.text

                                # Add AI's new response to chat history
                                st.session_state.chat_history.append({"role": "model", "parts": [{"text": response_text}]})

                                st.session_state.chat_input_value = "" # Clear input box
                                st.rerun() # Rerun to update chat display
                            except Exception as e:
                                st.error(f"❌ Error in chat: {e}")
                                st.warning("Could not get a response from AI Advisor. Please try again.")
                    else:
                        st.warning("Please type a message to send.")

# Always display the disclaimer at the bottom
st.markdown('<p class="disclaimer"><i class="fas fa-info-circle"></i> Disclaimer: This AI tool provides general information and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any health-related decisions. Your health is important!</p>', unsafe_allow_html=True)