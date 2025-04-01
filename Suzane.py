# import streamlit as st
# import whisper
# import nltk
# from sentence_transformers import SentenceTransformer
# import chromadb
# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings

# # Initialize the models
# whisper_model = whisper.load_model("base")
# nltk.download('punkt')
# sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
# client = chromadb.Client()
# collection = client.create_collection("subtitles_search")

# # Helper functions
# def transcribe_audio(audio_path):
#     result = whisper_model.transcribe(audio_path)
#     st.write(result['text'])
#     return result['text']

# def chunk_text(transcribed_text, chunk_size=500):
#     sentences = nltk.sent_tokenize(transcribed_text)
#     chunks = []
#     current_chunk = []
#     for sentence in sentences:
#         if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
#     return chunks

# def generate_embeddings(chunks):
#     return sentence_transformer.encode(chunks, convert_to_tensor=True)

# def store_embeddings_in_chroma(chunks, embeddings):
#     for i, chunk in enumerate(chunks):
#         collection.add(
#             documents=[chunk],
#             metadatas=[{"chunk_index": i}],
#             embeddings=[embeddings[i].cpu().numpy()]
#         )

# def search_subtitles(query):
#     embedding_function = SentenceTransformerEmbeddings(sentence_transformer)
#     query_embedding = embedding_function.embed(query)
#     chroma = Chroma(client=client, collection_name="subtitles_search")
#     results = chroma.similarity_search(query_embedding, k=5)
#     return results

# # Streamlit UI
# st.title("Audio Subtitle Search Engine")
# st.write("Upload an audio file to transcribe, chunk, vectorize and search subtitles.")

# # File upload
# audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

# if audio_file:
#     # Save uploaded audio
#     with open("uploaded_audio.mp3", "wb") as f:
#         f.write(audio_file.getbuffer())

#     # Step 1: Transcribe the audio
#     st.write("Transcribing the audio...")
#     transcribed_text = transcribe_audio("uploaded_audio.mp3")
#     st.write("Transcription Complete!")

#     # Step 2: Chunk the transcribed text
#     st.write("Chunking the transcription...")
#     chunks = chunk_text(transcribed_text, chunk_size=500)
#     st.write(f"Text has been chunked into {len(chunks)} pieces.")

#     # Step 3: Generate embeddings for each chunk
#     st.write("Generating embeddings for the chunks...")
#     embeddings = generate_embeddings(chunks)
#     st.write("Embeddings generated.")

#     # Step 4: Store embeddings in Chroma
#     store_embeddings_in_chroma(chunks, embeddings)
#     st.write("Embeddings stored in the database.")

#     # Step 5: Search functionality
#     query = st.text_input("Search for a subtitle segment:")
    
#     if query:
#         st.write("Searching for the query in the subtitles...")
#         results = search_subtitles(query)
        
#         if results:
#             st.write("Results:")
#             for result in results:
#                 st.write(f"- {result['document']}")
#         else:
#             st.write("No matching results found.")



# import streamlit as st
# import whisper
# import nltk
# from sentence_transformers import SentenceTransformer
# import chromadb
# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# import os

# # Ensure FFmpeg is available
# import subprocess
# os.environ["PATH"] += os.pathsep + "C:\\path\\to\\ffmpeg\\bin"  # Update this with your FFmpeg path

# try:
#     subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
#     st.write("FFmpeg is installed and available.")
# except subprocess.CalledProcessError:
#     st.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg.")

# # Initialize the models
# whisper_model = whisper.load_model("base")
# nltk.download('punkt')
# sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# # Ensure ChromaDB directory exists
# DB_PATH = "./chroma_db"
# if not os.path.exists(DB_PATH):
#     os.makedirs(DB_PATH)

# # Initialize ChromaDB Persistent Client
# client = chromadb.PersistentClient(path=DB_PATH)
# collection = client.get_or_create_collection("subtitles_search")

# st.write("ChromaDB Client Initialized Successfully!")

# # Helper functions
# def transcribe_audio(audio_path):
#     result = whisper_model.transcribe(audio_path)
#     st.write(result['text'])
#     return result['text']

# def chunk_text(transcribed_text, chunk_size=500):
#     sentences = nltk.sent_tokenize(transcribed_text)
#     chunks = []
#     current_chunk = []
#     for sentence in sentences:
#         if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
#     return chunks

# def generate_embeddings(chunks):
#     return sentence_transformer.encode(chunks, convert_to_tensor=True)

# def store_embeddings_in_chroma(chunks, embeddings):
#     for i, chunk in enumerate(chunks):
#         collection.add(
#             documents=[chunk],
#             metadatas=[{"chunk_index": i}],
#             embeddings=[embeddings[i].cpu().numpy()]
#         )

# def search_subtitles(query):
#     embedding_function = SentenceTransformerEmbeddings(sentence_transformer)
#     query_embedding = embedding_function.embed(query)
#     chroma = Chroma(client=client, collection_name="subtitles_search")
#     results = chroma.similarity_search(query_embedding, k=5)
#     return results

# # Streamlit UI
# st.title("Audio Subtitle Search Engine")
# st.write("Upload an audio file to transcribe, chunk, vectorize and search subtitles.")

# # File upload
# audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

# if audio_file:
#     # Save uploaded audio
#     with open("uploaded_audio.mp3", "wb") as f:
#         f.write(audio_file.getbuffer())

#     # Step 1: Transcribe the audio
#     st.write("Transcribing the audio...")
#     transcribed_text = transcribe_audio("uploaded_audio.mp3")
#     st.write("Transcription Complete!")

#     # Step 2: Chunk the transcribed text
#     st.write("Chunking the transcription...")
#     chunks = chunk_text(transcribed_text, chunk_size=500)
#     st.write(f"Text has been chunked into {len(chunks)} pieces.")

#     # Step 3: Generate embeddings for each chunk
#     st.write("Generating embeddings for the chunks...")
#     embeddings = generate_embeddings(chunks)
#     st.write("Embeddings generated.")

#     # Step 4: Store embeddings in Chroma
#     store_embeddings_in_chroma(chunks, embeddings)
#     st.write("Embeddings stored in the database.")

#     # Step 5: Search functionality
#     query = st.text_input("Search for a subtitle segment:")
    
#     if query:
#         st.write("Searching for the query in the subtitles...")
#         results = search_subtitles(query)
        
#         if results:
#             st.write("Results:")
#             for result in results:
#                 st.write(f"- {result['document']}")
#         else:
#             st.write("No matching results found.")






# import streamlit as st
# import whisper
# import nltk
# from sentence_transformers import SentenceTransformer
# import sqlite3
# import os
# import subprocess

# # Ensure FFmpeg is available
# def is_ffmpeg_installed():
#     try:
#         subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return False

# if not is_ffmpeg_installed():
#     st.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg and restart the application.")

# # Initialize models
# whisper_model = whisper.load_model("base")
# nltk.download('punkt')
# sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# # Database connection
# DB_PATH = "C:\\Users\\vmmeh\\Downloads\\eng_subtitles_database.db"  # Replace with your actual database path
# TABLE_NAME = "zipfiles"
# COLUMN_NAME = "content"

# def connect_db():
#     return sqlite3.connect(DB_PATH)

# def search_subtitles(query):
#     conn = connect_db()
#     cursor = conn.cursor()
#     cursor.execute(f"""
#         SELECT {COLUMN_NAME} FROM {TABLE_NAME}
#         WHERE {COLUMN_NAME} LIKE ?
#     """, (f'%{query}%',))
#     results = cursor.fetchall()
#     conn.close()
#     return [row[0] for row in results]

# # Helper functions
# def transcribe_audio(audio_path):
#     result = whisper_model.transcribe(audio_path)
#     st.write(result['text'])
#     return result['text']

# def chunk_text(transcribed_text, chunk_size=500):
#     sentences = nltk.sent_tokenize(transcribed_text)
#     chunks = []
#     current_chunk = []
#     for sentence in sentences:
#         if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
#     return chunks

# # Streamlit UI
# st.title("Audio Subtitle Search Engine")
# st.write("Upload an audio file to transcribe and search subtitles.")

# # File upload
# audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

# if audio_file:
#     # Save uploaded audio
#     with open("uploaded_audio.mp3", "wb") as f:
#         f.write(audio_file.getbuffer())

#     # Step 1: Transcribe the audio
#     st.write("Transcribing the audio...")
#     transcribed_text = transcribe_audio("uploaded_audio.mp3")
#     st.write("Transcription Complete!")

#     # Step 2: Chunk the transcribed text
#     st.write("Chunking the transcription...")
#     chunks = chunk_text(transcribed_text, chunk_size=500)
#     st.write(f"Text has been chunked into {len(chunks)} pieces.")

#     # Step 3: Search functionality
#     query = st.text_input("Search for a subtitle segment:")
    
#     if query:
#         st.write("Searching for the query in the database...")
#         results = search_subtitles(query)
        
#         if results:
#             st.write("Results:")
#             for result in results:
#                 st.write(f"- {result}")
#         else:
#             st.write("No matching results found.")



# import streamlit as st
# import whisper
# import os
# import glob
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def load_whisper_model():
#     return whisper.load_model("base")

# def transcribe_audio(video_path, model):
#     result = model.transcribe(video_path)
#     return result["text"]

# def load_subtitles(subtitle_folder):
#     subtitles = []
#     subtitle_files = glob.glob(os.path.join(subtitle_folder, "*.srt"))
    
#     for file in subtitle_files:
#         with open(file, "r", encoding="utf-8") as f:
#             subtitles.append(f.read())
    
#     return subtitles, subtitle_files

# def train_model(subtitle_folder):
#     subtitles, subtitle_files = load_subtitles(subtitle_folder)
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     documents = []
#     for subtitle in subtitles:
#         documents.extend(text_splitter.split_text(subtitle))
    
#     embeddings = HuggingFaceEmbeddings()
#     db = Chroma.from_texts(documents, embeddings)
#     return db

# def find_similar_subtitle(transcribed_text, db):
#     results = db.similarity_search(transcribed_text, k=1)
#     return results[0].page_content if results else "No matching subtitle found."

# # Streamlit UI
# st.title("Video Subtitle Search using AI")

# uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mkv"])

# if "whisper_model" not in st.session_state:
#     st.session_state.whisper_model = load_whisper_model()

# if "transcribed_text" not in st.session_state:
#     st.session_state.transcribed_text = ""

# if "best_subtitle" not in st.session_state:
#     st.session_state.best_subtitle = ""

# if "db" not in st.session_state:
#     st.session_state.db = train_model("subtitles2")  # Change path as needed

# if uploaded_file is not None:
#     st.video(uploaded_file)
#     video_path = os.path.join("temp_video.mp4")
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     if st.button("Transcribe Video"):
#         st.session_state.transcribed_text = transcribe_audio(video_path, st.session_state.whisper_model)
#         st.write("### Transcribed Text:")
#         st.write(st.session_state.transcribed_text)
    
#     if st.button("Search Similar Subtitle"):
#         st.session_state.best_subtitle = find_similar_subtitle(st.session_state.transcribed_text, st.session_state.db)
#         st.write("### Best Matching Subtitle:")
#         st.write(st.session_state.best_subtitle)
    
#     if st.button("Clear"):
#         st.session_state.transcribed_text = ""
#         st.session_state.best_subtitle = ""
#         st.experimental_rerun()


# from langchain_community.document_loaders import DirectoryLoader,TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# loader = DirectoryLoader("subtitles2",glob="*.srt",show_progress=True,loader_cls=TextLoader)

# docs = loader.load()

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

# chunks=text_splitter.split_documents(docs)

# print("No. Docs",len(docs))
# print("No. of chunks:",len(chunks))

# from langchain_chroma import Chroma
# # from suzane_chroma import docs       
# context_text="\n\n".join([docs[1].page_content for docs_score in doc_chroma])

# from langchain_core.prompts import ChatPromptTemplate

# PROMPT_TEMPLATE="""

# Give the similar subtitle from knowlede base for below context: {context}
# Find the similar subtitle on above context:{question}
# Provide a detailed answer. don't justify your answer.
# don't give information not mentioned in the context_information.
# do not say "according to the text" or "mentioned in the context" or similar.
# """

# prompt_template=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# query = "What is the best subtitle for this context?"
# prompt=prompt_template.format(context=context_text,question=query)



# from langchain_google_genai import ChatGoogleGenerativeAI

# file=open("api_suzane.txt","r")
# key=file.read().strip()
# model=ChatGoogleGenerativeAI(api_key=key,model="gemini-2.0-flash-exp")


# from langchain_core.output_parsers import StrOutputParser

# output_parser=StrOutputParser()

# chain = prompt_template|model|output_parser

# import streamlit as st
# import whisper
# import tempfile
# import os

# os.environ["PATH"] += os.pathsep + r"C:\Users\vmmeh\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-full_build\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin"


# def transcribe_audio(video_path):
#     model = whisper.load_model("base")  # Load Whisper model
#     result = model.transcribe(video_path)
#     return result["text"]

# st.title("🎤 Video to Text Transcription")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# if uploaded_file:
#     st.video(uploaded_file)  # Show uploaded video
    
#     if st.button("Transcribe"):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_path = tmp_file.name

#         st.write("Transcribing... ⏳")
#         transcript = transcribe_audio(tmp_path)
#         os.remove(tmp_path)  # Cleanup temporary file

#         st.text_area("Transcribed Text", transcript, height=200)
    
#     if st.button("Search subtitles"):
#         input_info={"context":context_text,"question":transcript}

#         response=chain.invoke(input_info)
        
#         st.text_area("subtitle", response, height=200)


# import os
# import tempfile
# import streamlit as st
# import whisper
# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Set the path for ChromaDB storage
# CHROMA_PATH = "D:\\PROGRAMS\\python\\streamlit\\myenv\\chroma_db2"
# file = open("api.txt", "r")
# key = file.read().strip()
# # Load Gemini embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)

# # Load the ChromaDB with Gemini embeddings
# vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# # Load Gemini model

# model = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash-exp")

# # Define the prompt template
# PROMPT_TEMPLATE = """
# Find the most similar subtitle from the knowledge base that closely matches the given transcribed text.  
# Use the provided context as reference to ensure the subtitle is semantically relevant.

# Context: {context}

# Transcribed Subtitle: {question}

# Return only the most relevant subtitle without any additional explanation.
# Provide a detailed answer. Don't justify your answer.
# Don't give information not mentioned in the context.
# Do not say 'according to the text' or 'mentioned in the context' or similar.
# """

# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# output_parser = StrOutputParser()
# chain = prompt_template | model | output_parser

# # Set FFMPEG Path
# os.environ["PATH"] += os.pathsep + r"C:\Users\vmmeh\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-full_build\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin"

# # Function to transcribe audio
# def transcribe_audio(video_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(video_path)
#     return result["text"]

# st.title("🎤 Video to Text Transcription & Subtitle Search")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
# transcript=''
# if uploaded_file:
#     st.video(uploaded_file)  # Show uploaded video
    
#     if st.button("Transcribe"):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_path = tmp_file.name

#         st.write("Transcribing... ⏳")
#         transcript = transcribe_audio(tmp_path)
#         os.remove(tmp_path)  # Cleanup temporary file

#         st.text_area("Transcribed Text", transcript, height=200)
    
#     if st.button("Search Subtitles"):
#         st.write("Searching subtitles... 🔍")
#         docs = retriever.get_relevant_documents(transcript)
#         context_text = "\n\n".join([doc.page_content for doc in docs])
        
#         input_info = {"context": context_text, "question": transcript}
#         response = chain.invoke(input_info)
        
#         st.text_area("Subtitle", response, height=200)


# import gradio as gr
# import json
# import whisper
# import chromadb
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from sentence_transformers import SentenceTransformer
# import tempfile
# import os

# # Set ChromaDB Path
# CHROMA_DB_PATH = r"D:\PROGRAMS\python\streamlit\myenv\chroma_db2"

# os.environ["PATH"] += os.pathsep + r"C:\Users\vmmeh\Downloads\ffmpeg-2025-03-31-git-35c091f4b7-full_build\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin"
# # Load ChromaDB Client & Collection
# client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# collection = client.get_or_create_collection(name="subtitle_chunks")

# # Load Gemini Embeddings
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="YOUR_GEMINI_API_KEY")

# # Load Whisper Model
# whisper_model = whisper.load_model("base")

# # Function to Transcribe Video
# def transcribe_video(video_file):
#     if not video_file:
#         return "Please upload a video file.", None
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#         tmp_file.write(video_file.read())
#         tmp_path = tmp_file.name

#     result = whisper_model.transcribe(tmp_path)
#     os.remove(tmp_path)  # Cleanup

#     return result["text"], result["text"]

# # Function to Format Search Results
# def format_results_as_json(results):
#     formatted_results = []
#     if results and results["metadatas"] and results["metadatas"][0]:
#         for i, metadata in enumerate(results["metadatas"][0]):
#             subtitle_name = metadata.get("subtitle_name", "Unknown")
#             subtitle_id = metadata.get("subtitle_id", "N/A")
#             url = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"
#             formatted_results.append({
#                 "Result": i + 1,
#                 "Subtitle Name": subtitle_name.upper(),
#                 "URL": url,
#             })
#         return json.dumps(formatted_results, indent=4)
#     return json.dumps([{"Result": "No results found"}], indent=4)

# # Function to Search Subtitles in ChromaDB
# def search_subtitles(query):
#     if not query:
#         return "No transcription text available for search."
    
#     # Generate Embedding
#     query_embedding = embedding_model.embed_query(query)
    
#     # Search ChromaDB
#     results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["metadatas"])
    
#     return format_results_as_json(results)

# # Function to Clear All Inputs
# def clear_all():
#     return "", ""

# # Gradio Interface with Styling
# custom_css = """
#     .gradio-container { width: 90% !important; margin: auto; }
#     button { padding: 10px; border-radius: 5px; cursor: pointer; transition: opacity 0.3s; }
#     .transcribe-btn { background-color: green; color: white; }
#     .transcribe-btn:hover { opacity: 0.8; }
#     .search-btn { background-color: blue; color: white; }
#     .search-btn:hover { opacity: 0.8; }
#     .clear-btn { background-color: #ff4500; color: white; }
#     .clear-btn:hover { opacity: 0.8; }
# """

# with gr.Blocks(css=custom_css) as demo:
#     gr.Markdown("# 🎬 Video Subtitle Search: Transcribe & Find Similar Subtitles", elem_id="title")
#     text_state = gr.State(value="")

#     with gr.Row():
#         video_input = gr.File(label="Upload Video", type="filepath")
#         transcribed_text = gr.Textbox(label="Transcribed Text", interactive=False)

#     with gr.Row():
#         transcribe_button = gr.Button("Transcribe", elem_classes=["transcribe-btn"])
#         search_button = gr.Button("Search Subtitles", elem_classes=["search-btn"])
#         clear_button = gr.Button("Clear", elem_classes=["clear-btn"])

#     search_results = gr.Textbox(label="Subtitle Search Results")

#     # Button Actions
#     transcribe_button.click(transcribe_video, inputs=[video_input], outputs=[transcribed_text, text_state])
#     search_button.click(search_subtitles, inputs=[text_state], outputs=[search_results])
#     clear_button.click(clear_all, inputs=[], outputs=[transcribed_text, search_results])

# # Launch App
# demo.launch()

# import os
# import tempfile
# import json
# import streamlit as st
# import whisper
# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Set the path for ChromaDB storage
# CHROMA_PATH = "D:\\PROGRAMS\\python\\streamlit\\myenv\\chroma_db2"
# file = open("api.txt", "r")
# key = file.read().strip()

# # Load Gemini embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)

# # Load the ChromaDB with Gemini embeddings
# vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# # Load Gemini model
# model = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash-exp")

# # Define the prompt template
# PROMPT_TEMPLATE = """
# Find the most relevant subtitle from the knowledge base that closely matches the given transcribed text.
# Use the provided context as reference to ensure the subtitle is semantically relevant.

# Context: {context}

# Transcribed Subtitle: {question}

# Return only the most relevant subtitle.
# """

# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# output_parser = StrOutputParser()
# chain = prompt_template | model | output_parser

# # Set FFMPEG Path
# os.environ["PATH"] += os.pathsep + r"C:\\Users\\vmmeh\\Downloads\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\bin"

# # Function to transcribe audio
# def transcribe_audio(video_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(video_path)
#     return result["text"]

# # Function to format subtitle results as JSON
# def format_results_as_json(subtitle_text, source_text):
#     """Formats subtitle results into JSON format with actual retrieved text."""
#     formatted_results = [
#         {
#             "Result": 1,
#             "Subtitle": subtitle_text,
#             "Source": source_text,
#         }
#     ]
#     return json.dumps(formatted_results, indent=4)

# st.title("🎤 Video to Text Transcription & Subtitle Search")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
# transcript = ""

# if uploaded_file:
#     st.video(uploaded_file)  # Show uploaded video
    
#     if st.button("Transcribe"):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_path = tmp_file.name

#         st.write("Transcribing... ⏳")
#         transcript = transcribe_audio(tmp_path)
#         os.remove(tmp_path)  # Cleanup temporary file

#         st.text_area("Transcribed Text", transcript, height=200)
    
#     if st.button("Search Subtitles"):
#         st.write("Searching subtitles... 🔍")
#         docs = retriever.get_relevant_documents(transcript)
        
#         if docs:
#             most_relevant_subtitle = docs[0].page_content  # Extract the most relevant subtitle from database
#         else:
#             most_relevant_subtitle = "No relevant subtitle found."
        
#         input_info = {"context": most_relevant_subtitle, "question": transcript}
#         response = chain.invoke(input_info)

#         # Format response as JSON
#         json_result = format_results_as_json(response, most_relevant_subtitle)
        
#         st.text_area("Subtitle (JSON Output)", json_result, height=200)


# import os
# import json
# import streamlit as st
# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document

# # Set the path for ChromaDB storage
# CHROMA_PATH = "D:\\PROGRAMS\\python\\streamlit\\myenv\\chroma_db2"
# file = open("api.txt", "r")
# key = file.read().strip()

# # Load Gemini embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)

# # Load the ChromaDB with Gemini embeddings
# vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# st.write(vectorstore._collection.count())
# # Load Gemini model
# model = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash-exp")

# # Define the prompt template
# PROMPT_TEMPLATE = """
# Find the most relevant subtitle from the knowledge base that closely matches the given text.
# Use the provided context as reference to ensure the subtitle is semantically relevant.

# Context: {context}

# Given Text: {question}

# Return only the most relevant subtitle.
# """

# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# output_parser = StrOutputParser()
# chain = prompt_template | model | output_parser

# # Function to format subtitle results as JSON
# def format_results_as_json(subtitle_text, source_text):
#     """Formats subtitle results into JSON format with actual retrieved text."""
#     formatted_results = [
#         {
#             "Result": 1,
#             "Subtitle": subtitle_text,
#             "Source": source_text,
#         }
#     ]
#     return json.dumps(formatted_results, indent=4)

# # Function to load subtitles into ChromaDB
# def load_subtitles():
#     subtitle_folder = "D:\\PROGRAMS\\python\\streamlit\\subtitle2"
#     subtitle_files = [f for f in os.listdir(subtitle_folder) if f.endswith(".srt")]
    
#     if not subtitle_files:
#         print("No subtitle files found.")
#         return
    
#     first_subtitle_file = os.path.join(subtitle_folder, subtitle_files[0])
#     with open(first_subtitle_file, "r", encoding="utf-8") as file:
#         subtitle_text = file.read()
    
#     docs = [Document(page_content=subtitle_text)]
#     vectorstore.add_documents(docs)
#     print("Loaded first subtitle into ChromaDB.")

# # Load subtitles into ChromaDB (only if empty)
# if vectorstore._collection.count() == 0:
#     load_subtitles()

# st.title("🎬 Movie Details Search")

# movie_name = st.text_input("Enter movie name:")
# if st.button("Search Movie"):
#     st.write("Searching movie details... 🔍")
#     docs = retriever.get_relevant_documents(movie_name)
    
#     if docs:
#         movie_details = docs[0].page_content  # Get the most relevant movie details
#     else:
#         movie_details = "No movie details found."
    
#     st.text_area("Movie Details", movie_details, height=200)




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

os.environ["PATH"] += os.pathsep + r"C:\\Users\\vmmeh\\Downloads\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\bin"
# Set the path for ChromaDB storage
CHROMA_PATH = "D:\\PROGRAMS\\python\\streamlit\\myenv\\chroma_db2"
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
    subtitle_folder = "D:\\PROGRAMS\\python\\streamlit\\subtitle2"
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
