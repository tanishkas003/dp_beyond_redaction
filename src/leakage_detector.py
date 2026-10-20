"""Transparent deterministic baseline leakage checker."""
import re
from dataclasses import dataclass
@dataclass(frozen=True)
class LeakageFinding: kind: str; start: int; end: int
@dataclass(frozen=True)
class LeakageReport: safe: bool; findings: tuple = ()
class LeakageDetector:
    PATTERNS=(("SSN",re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),("EMAIL",re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),("API_KEY",re.compile(r"\b(?:sk|api)[_-][A-Za-z0-9_-]{16,}\b",re.I)),("PRIVATE_KEY",re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),("PASSWORD",re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*\S+")))
    def check(self,text,protected_entities=()):
        findings=[LeakageFinding(kind,m.start(),m.end()) for kind,p in self.PATTERNS for m in p.finditer(text)]
        for entity in protected_entities:
            raw=getattr(entity,"original",None)
            if raw and raw in text:
                start=text.find(raw); findings.append(LeakageFinding("RESIDUAL_ENTITY",start,start+len(raw)))
        unique={(x.kind,x.start,x.end):x for x in findings}; return LeakageReport(not unique,tuple(unique.values()))
