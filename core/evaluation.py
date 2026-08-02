import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings

class RAGEvaluator:
    def __init__(self):
        self.judge_llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL_NAME,
            temperature=0.1
        )
        self._build_evaluator()
    
    def _build_evaluator(self):
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. You need to check if the given answer is FACTUALLY ACCURATE based on the provided context.
        
        Context: {context}
        Answer: {answer}
        
        Evaluate the answer on the following criteria:
        1. Faithfulness: Is the answer fully based on the provided context? (Yes/No)
        2. Hallucination: Did the model create any false information not in the context? (Yes/No)
        3. Confidence Score: Give a score from 0 to 1 (1 = completely faithful)
        
        Respond ONLY with a JSON:
        {{
            "faithful": true/false,
            "hallucinated": true/false,
            "confidence": 0.95
        }}
        """)
        
        self.eval_chain = prompt | self.judge_llm | StrOutputParser()
    
    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """Evaluate answer using LLM-as-Judge"""
        
        result = self.eval_chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })
        
        try:
            cleaned = result.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except:
            return {
                "faithful": False,
                "hallucinated": True,
                "confidence": 0.0,
                "error": result
            }
    
    def is_answer_valid(self, eval_result: dict) -> bool:
        """Check if answer passes quality gate"""
        threshold = settings.CONFIDENCE_THRESHOLD
        return (eval_result.get("faithful", False) and 
                not eval_result.get("hallucinated", True) and
                eval_result.get("confidence", 0) >= threshold)