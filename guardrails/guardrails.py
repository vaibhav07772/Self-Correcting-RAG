import re
from typing import Dict, Any

class InputGuardrail:
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b'
        }
        self.injection_patterns = [
            "ignore previous", "forget your training", "system prompt",
            "you are now", "role:", "pretend you are", "jailbreak"
        ]
    
    def detect_prompt_injection(self, text: str) -> bool:
        for pattern in self.injection_patterns:
            if pattern.lower() in text.lower():
                return True
        return False
    
    def redact_pii(self, text: str) -> str:
        for name, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[REDACTED_{name.upper()}]", text)
        return text
    
    def process(self, text: str) -> Dict[str, Any]:
        is_safe = True
        warnings = []
        
        if self.detect_prompt_injection(text):
            is_safe = False
            warnings.append("Prompt injection detected")
        
        redacted = self.redact_pii(text)
        
        return {
            "safe": is_safe,
            "redacted": redacted,
            "warnings": warnings
        }

input_guardrail = InputGuardrail()