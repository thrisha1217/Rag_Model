import streamlit as st
from openai import OpenAI
import requests # Keep requests if used elsewhere in RAG, but it's not needed for OpenAI API calls

# -----------------------------------------------------------
# Caching the OpenAI client for efficiency
# @st.cache_resource ensures the client is initialized only once
# -----------------------------------------------------------
@st.cache_resource
def get_openai_client():
    # 1. Retrieve API key from Streamlit secrets
    api_key = st.secrets.get("OPENAI_API_KEY") 
    
    if not api_key:
        # Halt execution if key is missing (Crucial for deployed apps)
        st.error("🔑 OPENAI_API_KEY secret not found. Please set it in Streamlit Cloud Secrets.")
        st.stop()
    
    # 2. Initialize the OpenAI client
    # The client automatically uses the API key
    client = OpenAI(api_key=api_key)
    return client

# -----------------------------------------------------------
# Function to generate the answer using the OpenAI API
# -----------------------------------------------------------
def generate_answer(prompt, model_name="gpt-3.5-turbo"):
    client = get_openai_client()
    
    # 3. Use the chat completions endpoint
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # Use a system message to guide the LLM's behavior
                {"role": "system", "content": "You are an accurate Retrieval-Augmented Generation (RAG) model. Base your answer strictly on the context provided in the user prompt."},
                # The prompt from the RAG system (context + question)
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 # Use low temperature for RAG to keep answers factual
        )
        
        # 4. Extract the answer text
        return response.choices[0].message.content
        
    except Exception as e:
        # Display API errors to the user
        st.error(f"An error occurred during LLM generation: {e}")
        return "Failed to generate an answer due the LLM API error.", ""


# In your main app.py file (around line 66):
# The call remains the same:
# answer, context = rag_response(question, embedder, index, id_to_text) 

# And rag_response (around line 53) remains the same:
# answer = generate_answer(prompt) 
# The magic happens inside generate_answer now!
