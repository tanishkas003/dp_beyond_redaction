"""Offline attacker interfaces; attackers never receive ground truth."""
import re
from dataclasses import dataclass
class DeterministicAttacker:
    name="deterministic-pattern"
    def predict(self,text,question=""): return re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|\b\d{3}-\d{2}-\d{4}\b",text)
@dataclass
class LLMAttacker:
    call: object; name: str="llm-attacker"
    def predict(self,text,question=""): return [x.strip() for x in self.call(f"Infer sensitive values without ground truth. Question: {question}\nText: {text}").splitlines() if x.strip()]
def attack_success_rate(attacker,protected_texts,truths,question=""):
    texts,expected=list(protected_texts),list(truths)
    if len(texts)!=len(expected): raise ValueError("inputs must have equal length")
    correct=0
    for text,truth in zip(texts,expected): correct += any(truth.casefold()==guess.casefold() for guess in attacker.predict(text,question))
    return {"attacker":attacker.name,"attempted":len(texts),"correct":correct,"attack_success_rate":correct/len(texts) if texts else 0.0}
