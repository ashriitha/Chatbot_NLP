import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL Fix for nltk downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to get chatbot response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Initialize session state for chat history and input tracking
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

# Streamlit UI Configuration
st.set_page_config(page_title="Financial AI Chatbot", layout="wide")
st.title("🤖 Financial AI Chatbot")
st.write("Welcome! Type your message below and start chatting.")

# Sidebar Menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Chat Interface")

    # Chat Display Container
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**You:** {chat['user']}")
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {chat['bot']}")

    # Single Input Box
    user_input = st.text_input("Type your message and press Enter:", key="user_input")

    # Process input only if it's different from the last input
    if user_input and user_input != st.session_state.last_input:
        st.session_state.last_input = user_input  # Update last input to prevent re-triggering

        response = chatbot(user_input)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})

        # Save chat history to CSV
        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

        # Clear input field by removing it from session state
        st.session_state.pop("user_input", None)

        # Refresh UI
        st.rerun()

    # Clear Chat Button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_input = ""  # Reset last input to prevent unwanted responses
        st.rerun()

elif choice == "Conversation History":
    st.header("📜 Conversation History")

    # Check if chat_log.csv exists
    if os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    else:
        st.write("No conversation history available.")

elif choice == "About":
    st.header("ℹ️ About the Chatbot")
    st.write("""
    This chatbot is designed to answer financial queries using NLP and Logistic Regression.
    
    **Features:**
    - Understands predefined intents based on training data.
    - Uses **TF-IDF Vectorization** and **Logistic Regression** for response prediction.
    - Provides a **clear chat button** to reset conversations.
    - Saves chat history in a CSV file for future reference.
    
    Built using **Streamlit** for an interactive UI.
    """)

# Run the app
if __name__ == '__main__':
    pass
