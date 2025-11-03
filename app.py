import streamlit as st
import requests
# Add the OpenAI library import
from openai import OpenAI  # If using OpenAI

# Initialize the client outside the function for caching benefits
@st.cache_resource
def get_openai_client():
    # Access the API key securely from Streamlit secrets
    api_key = st.secrets.get("OPENAI_API_KEY") 
    if not api_key:
        st.error("OPENAI_API_KEY secret not found. Please set it in Streamlit Secrets.")
        return None
    
    # Initialize the client
    client = OpenAI(api_key=api_key)
    return client

def generate_answer(prompt, model_name="gpt-3.5-turbo"):
    # 🚨 DELETE the old Ollama code:
    # response = requests.post(
    #    "http://localhost:11434/api/generate",
    #    json={"model": model_name, "prompt": prompt}
    # )

    client = get_openai_client()
    if not client:
        return "Error: LLM client failed to initialize.", ""
        
    try:
        # Use the OpenAI API for chat completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful RAG model."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An API error occurred: {e}", ""

# ... rest of your Streamlit code ...
