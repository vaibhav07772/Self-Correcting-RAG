import os
import sys
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- Fix: Add project root to path ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

class DocumentIndexer:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def load_documents(self, file_paths: List[str]) -> List:
        """Load PDF, TXT files"""
        documents = []
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            elif path.endswith('.txt'):
                loader = TextLoader(path)
                documents.extend(loader.load())
        return documents
    
    def ingest(self, file_paths: List[str]):
        """Full ingestion pipeline"""
        # Step 1: Load
        docs = self.load_documents(file_paths)
        
        # Step 2: Split
        chunks = self.text_splitter.split_documents(docs)
        
        # Step 3: Embeddings + Vector Store
        vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        
        print(f"✅ Ingested {len(chunks)} chunks from {len(file_paths)} files")
        return vectorstore

if __name__ == "__main__":
    # Check if data folder exists
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("📁 Created 'data' folder. Please add your PDF/TXT files there.")
    else:
        indexer = DocumentIndexer()
        files = [f"./data/{f}" for f in os.listdir("./data") if f.endswith(('.pdf', '.txt'))]
        if files:
            indexer.ingest(files)
        else:
            print("⚠️ No PDF/TXT files found in 'data' folder. Please add some.")