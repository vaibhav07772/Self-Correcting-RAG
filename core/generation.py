import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from core.retrieval import HybridRetriever

class RAGGenerator:
    def __init__(self):
        self.retriever = HybridRetriever().get_retriever()
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL_NAME,
            temperature=0.3
        )
        self._build_chain()
    
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based ONLY on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate(self, question: str) -> dict:
        """Generate answer with context"""
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        
        context = "\n".join([doc.page_content for doc in docs[:3]])
        
        return {
            "answer": answer,
            "context": context,
            "sources": [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs[:3]],
            "num_sources": len(docs)
        }