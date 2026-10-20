"""Configurable, explainable per-entity privacy risk assessment."""
from dataclasses import dataclass
from enum import Enum
import re

class RiskCategory(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"; CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class RiskWeights:
    sensitivity: float = .28; identifiability: float = .18; uniqueness: float = .14
    reidentification: float = .18; recoverability: float = .10; task_requirement: float = .12

@dataclass(frozen=True)
class RiskThresholds:
    medium: float = .30; high: float = .60; critical: float = .85

@dataclass(frozen=True)
class RiskFactors:
    sensitivity: float; identifiability: float; uniqueness: float
    reidentification: float; recoverability: float; task_requirement: float
    def as_dict(self): return vars(self).copy()

@dataclass(frozen=True)
class RiskAssessment:
    entity_type: str; risk_score: float; category: RiskCategory; factors: RiskFactors
    exact_recovery_required: bool; recommendation: str; reasons: tuple = (); critical: bool = False

class PrivacyRiskEngine:
    _BASE = {
        "SSN": (.98,.98,.95,.98), "CREDIT_CARD": (.98,.98,.95,.98),
        "PASSWORD": (1,1,1,1), "PRIVATE_KEY": (1,1,1,1), "API_KEY": (1,1,1,1),
        "AUTH_TOKEN": (1,1,1,1), "EMAIL": (.78,.86,.72,.78), "PHONE": (.74,.82,.70,.72),
        "PERSON": (.70,.74,.65,.70), "EMPLOYEE_ID": (.76,.90,.78,.78),
        "LOCATION": (.45,.56,.45,.48), "ORG": (.42,.55,.42,.45),
        "JOB_TITLE": (.32,.35,.25,.28), "DATE": (.38,.42,.38,.35),
        "AGE": (.55,.60,.52,.52), "FINANCIAL_AMOUNT": (.84,.76,.80,.86),
        "MEDICAL": (.92,.80,.82,.92),
    }
    _CRITICAL = {"PASSWORD","PRIVATE_KEY","API_KEY","API_SECRET","AUTH_TOKEN","CREDENTIAL"}
    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or RiskWeights(); self.thresholds = thresholds or RiskThresholds()
    def assess(self, entity_text, entity_type, context="", task_requirements=None, features=None):
        del entity_text
        kind = (entity_type or "UNKNOWN").upper().strip().replace(" ", "_")
        req = task_requirements or {}; base = self._BASE.get(kind, (.72,.70,.70,.75))
        vals = dict(zip(("sensitivity","identifiability","uniqueness","reidentification"), base))
        for key in vals:
            if features and key in features: vals[key] = min(max(float(features[key]),0),1)
        words = set(re.findall(r"[a-z0-9-]+", (context or "").lower()))
        exact = bool(req.get("exact_recovery", bool(words & {"exact","original","verbatim","unchanged","precise"}) or kind in {"SSN","EMAIL","PHONE","PERSON","EMPLOYEE_ID","PROJECT_CODE","INVOICE_NO"}))
        factors = RiskFactors(**vals, recoverability=float(req.get("recoverability", 1 if exact else vals["reidentification"])), task_requirement=float(req.get("task_requirement", 1 if exact else .35)))
        weights = vars(self.weights); score = sum(getattr(factors,k)*weights[k] for k in weights)/sum(weights.values())
        critical = kind in self._CRITICAL
        if critical or score >= self.thresholds.critical: category, rec = RiskCategory.CRITICAL, "BLOCK_OR_LOCAL"
        elif score >= self.thresholds.high: category, rec = RiskCategory.HIGH, "LOCAL_OR_APPROVED_POLICY"
        elif score >= self.thresholds.medium: category, rec = RiskCategory.MEDIUM, "GENERALIZE_OR_EXPONENTIAL_MECHANISM"
        else: category, rec = RiskCategory.LOW, "CLOUD_AFTER_LEAKAGE_CHECK"
        return RiskAssessment(kind, round(score,6), category, factors, exact, rec, ("critical-data policy override",) if critical else (), critical)
