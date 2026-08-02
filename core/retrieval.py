import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config.settings import settings

class HybridRetriever:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self._load_retriever()
    
    def _load_retriever(self):
        # Simple Vector Retriever
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def get_retriever(self):
        return self.retriever