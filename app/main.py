import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from core.self_correction import SelfCorrectingRAG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Correcting RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SelfCorrectingRAG(max_attempts=3)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    attempts: int
    is_valid: bool
    confidence: float
    hallucinated: bool
    sources: List = []

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Self-Correcting RAG"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info(f"📝 Query: {request.question}")
    
    try:
        result = rag.query(request.question)
        
        return QueryResponse(
            answer=result["final_answer"],
            attempts=result["attempts"],
            is_valid=result["is_valid"],
            confidence=result.get("eval_scores", {}).get("confidence", 0.0),
            hallucinated=result.get("eval_scores", {}).get("hallucinated", True),
            sources=[]
        )
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))