


import os
import tempfile
import json
import streamlit as st
import whisper
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


# Set the path for ChromaDB storage
CHROMA_PATH = "chroma_db2"
file = open("api.txt", "r")
key = file.read().strip()

# Load Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)

# Load the ChromaDB with Gemini embeddings
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Load Gemini model
model = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash-exp")

# Define the prompt template
PROMPT_TEMPLATE = """
Find the most relevant subtitle from the knowledge base that closely matches the given text.
Use the provided context as reference to ensure the subtitle is semantically relevant.

Context: {context}

Given Text: {question}

Return only the most relevant subtitle.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
output_parser = StrOutputParser()
chain = prompt_template | model | output_parser

# Function to format subtitle results as JSON
def format_results_as_json(subtitle_text, source_text):
    """Formats subtitle results into JSON format with actual retrieved text."""
    formatted_results = [
        {
            "Result": 1,
            "Subtitle": subtitle_text,
            "Source": source_text,
        }
    ]
    return json.dumps(formatted_results, indent=4)

# Function to load subtitles into ChromaDB
def load_subtitles():
    subtitle_folder = "subtitle2"
    subtitle_files = [f for f in os.listdir(subtitle_folder) if f.endswith(".srt")]
    
    if not subtitle_files:
        print("No subtitle files found.")
        return
    
    first_subtitle_file = os.path.join(subtitle_folder, subtitle_files[0])
    with open(first_subtitle_file, "r", encoding="utf-8") as file:
        subtitle_text = file.read()
    
    docs = [Document(page_content=subtitle_text)]
    vectorstore.add_documents(docs)
    print("Loaded first subtitle into ChromaDB.")

# Load subtitles into ChromaDB (only if empty)
if vectorstore._collection.count() == 0:
    load_subtitles()

st.title("🎬 Video to Text Transcription & Subtitle Search")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
transcript = ""

if uploaded_file:
    st.video(uploaded_file)  # Show uploaded video
    
    if st.button("Transcribe"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.write("Transcribing... ⏳")
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        transcript = result["text"]
        os.remove(tmp_path)  # Cleanup temporary file

        st.text_area("Transcribed Text", transcript, height=200)
    
    if st.button("Search Subtitles"):
        st.write("Searching subtitles... 🔍")
        docs = retriever.get_relevant_documents(transcript)
        
        if docs:
            most_relevant_subtitle = docs[0].page_content  # Extract the most relevant subtitle from database
        else:
            most_relevant_subtitle = "No relevant subtitle found."
        
        input_info = {"context": most_relevant_subtitle, "question": transcript}
        response = chain.invoke(input_info)

        # Format response as JSON
        json_result = format_results_as_json(response, most_relevant_subtitle)
        
        st.text_area("Subtitle (JSON Output)", json_result, height=200)
