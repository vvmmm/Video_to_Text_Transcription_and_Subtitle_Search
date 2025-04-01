from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = DirectoryLoader(
    "subtitles2",
    glob="*.srt",
    show_progress=True,
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")  # Force UTF-8 encoding
)


docs = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

chunks=text_splitter.split_documents(docs)

print("No. Docs",len(docs))
print("No. of chunks:",len(chunks))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

file=open("api.txt","r")
key=file.read().strip()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=key  # API Key Directly in Code
)



from langchain_chroma import Chroma

db=Chroma(collection_name="vector_database",embedding_function=embeddings,persist_directory="./chroma_db2")

batch_size=166
for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        db.add_documents(batch)
        print(f"✅ Inserted batch {i//batch_size + 1}")