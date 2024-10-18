import streamlit as st
import google.generativeai as genai
import os
from streamlit_chat import message

# Set up Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get response from Gemini
def get_bot_response(prompt, history):
    # Convert history to the format Gemini expects
    gemini_history = [
        {"role": "user" if msg["role"] == "human" else "model", "parts": [msg["content"]]}
        for msg in history
    ]
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Set page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for Claude-like UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        caret-color: #9D5CFF;
    }
    .stButton > button {
        background-color: #9D5CFF;
        color: white;
    }
    .stButton > button:hover {
        background-color: #7B3FCC;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("This chatbot uses Google's Gemini model to generate responses.")
    data_link = "https://huggingface.co/datasets/your_dataset"
    st.markdown(f"[Link to the dataset]({data_link})")

# Main content
st.title("AI Chatbot")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display chat messages
for i, chat in enumerate(st.session_state['history']):
    if chat['role'] == 'human':
        message(chat['content'], is_user=True, key=f"{i}_user")
    else:
        message(chat['content'], is_user=False, key=f"{i}_bot")

# Chat input
with st.container():
    user_input = st.text_input("You:", key="user_input")
    send_button = st.button("Send")

if send_button and user_input:
    # Add user input to history
    st.session_state['history'].append({"role": "human", "content": user_input})
    
    # Get bot response
    bot_response = get_bot_response(user_input, st.session_state['history'])
    
    # Add bot response to history
    st.session_state['history'].append({"role": "ai", "content": bot_response})
    
    # Clear input
    st.session_state['user_input'] = ""
    
    # Rerun to update chat display
    st.experimental_rerun()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state['history'] = []
    st.experimental_rerun()
