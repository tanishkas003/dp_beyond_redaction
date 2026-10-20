from dataclasses import dataclass
from src.leakage_detector import LeakageDetector
@dataclass(frozen=True)
class ResponseFilterResult: allowed: bool; response: str; report: object
class ResponsePrivacyFilter:
    def __init__(self,detector=None): self.detector=detector or LeakageDetector()
    def filter(self,response,protected_entities=()):
        report=self.detector.check(response,protected_entities); return ResponseFilterResult(report.safe,response if report.safe else "",report)
