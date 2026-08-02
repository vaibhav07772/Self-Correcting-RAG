import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from core.generation import RAGGenerator
from core.evaluation import RAGEvaluator

# State definition
class AgentState(TypedDict):
    question: str
    answer: str
    context: str
    eval_result: dict
    is_valid: bool
    attempts: int
    max_attempts: int
    final_answer: str

class SelfCorrectingRAG:
    def __init__(self, max_attempts: int = 3):
        self.generator = RAGGenerator()
        self.evaluator = RAGEvaluator()
        self.max_attempts = max_attempts
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("correct", self._correct_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge: if invalid and attempts < max → correct → generate
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "correct": "correct",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("correct", "generate")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _generate_node(self, state: AgentState) -> AgentState:
        result = self.generator.generate(state["question"])
        state["answer"] = result["answer"]
        state["context"] = result["context"]
        return state
    
    def _evaluate_node(self, state: AgentState) -> AgentState:
        eval_result = self.evaluator.evaluate(
            state["question"],
            state["answer"],
            state["context"]
        )
        state["eval_result"] = eval_result
        state["is_valid"] = self.evaluator.is_answer_valid(eval_result)
        return state
    
    def _correct_node(self, state: AgentState) -> AgentState:
        correction_prompt = f"""
        Your previous answer was incorrect. The evaluator found the following issues:
        {state['eval_result']}
        
        Please provide a corrected answer based ONLY on the provided context.
        """
        state["question"] = correction_prompt + "\n\nOriginal question: " + state["question"]
        state["attempts"] = state.get("attempts", 0) + 1
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        state["final_answer"] = state["answer"]
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        if (state["is_valid"] or 
            state.get("attempts", 0) >= self.max_attempts):
            return "finalize"
        return "correct"
    
    def query(self, question: str) -> dict:
        """Main entry point"""
        initial_state = {
            "question": question,
            "answer": "",
            "context": "",
            "eval_result": {},
            "is_valid": False,
            "attempts": 0,
            "max_attempts": self.max_attempts,
            "final_answer": ""
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "final_answer": result.get("final_answer", "Unable to generate answer"),
            "attempts": result.get("attempts", 0),
            "is_valid": result.get("is_valid", False),
            "eval_scores": result.get("eval_result", {})
        }