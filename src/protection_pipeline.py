"""Integrated detection, policy, protection, provider, and restoration flow."""
from dataclasses import dataclass, field
from typing import Optional
from src.critical_policy import CriticalDataPolicy
from src.detection import DetectedEntity
from src.dp_noise import exponential_sanitize, sanitize_entity
from src.generalization import generalize as semantic_generalize
from src.leakage_detector import LeakageDetector
from src.mechanism_router import MechanismRouter, Mechanism
from src.model_router import ModelRouter, ProviderUnavailable
from src.privacy_budget import PrivacyBudgetAccountant, PrivacyBudgetExceeded
from src.response_filter import ResponsePrivacyFilter
from src.risk_engine import PrivacyRiskEngine, RiskCategory
from src.vault import SessionVault
from src.utility_assessor import UtilityAssessor, extract_features

class ProtectionBlocked(RuntimeError): pass
@dataclass
class ProtectedEntity:
    original: str; surrogate: str; entity_type: str; mechanism: str; start: int; end: int; risk: object = None
@dataclass
class ProtectionResult:
    original_text: str; protected_text: str; protected_entities: list[ProtectedEntity]
    assessments: list = field(default_factory=list); policy: object = None; leakage_report: object = None
    blocked: bool = False; metadata: dict = field(default_factory=dict)

class ProtectionPipeline:
    def __init__(self, detector, router: MechanismRouter, vault: SessionVault, tokenizer=None, dp_mechanism=None, epsilon=1.0,
                 risk_engine=None, budget=None, critical_policy=None, leakage_detector=None, response_filter=None,
                 model_router=None, utility_assessor=None, embed_fn=None,
                 max_leakage_retries=1, use_exponential_mechanism=True):
        del tokenizer
        self.detector=detector; self.router=router; self.vault=vault; self.dp_mechanism=dp_mechanism; self.epsilon=epsilon
        self.risk_engine=risk_engine or PrivacyRiskEngine(); self.budget=budget or PrivacyBudgetAccountant()
        self.critical_policy=critical_policy or CriticalDataPolicy(); self.leakage_detector=leakage_detector or LeakageDetector()
        self.response_filter=response_filter or ResponsePrivacyFilter(self.leakage_detector); self.model_router=model_router
        self.utility_assessor = utility_assessor
        self.embed_fn = embed_fn
        self.max_leakage_retries=max(0,int(max_leakage_retries)); self.use_exponential_mechanism=use_exponential_mechanism
    @staticmethod
    def _non_overlapping(entities):
        selected=[]
        for entity in sorted(entities,key=lambda x:(x.start,-(x.end-x.start))):
            if not any(entity.start < old.end and entity.end > old.start for old in selected): selected.append(entity)
        return selected
    def protect(self,text,task_requirements=None):
        if not isinstance(text,str): raise TypeError("prompt must be text")
        entities=self._non_overlapping(self.detector.detect(text)); policy=self.critical_policy.inspect(text,{e.entity_type for e in entities})
        if not policy.allowed: raise ProtectionBlocked("prompt blocked by critical-data policy")
        protected=text; records=[]; assessments=[]
        for entity in sorted(entities,key=lambda x:x.start,reverse=True):
            assessment=self.risk_engine.assess(entity.text,entity.entity_type,text,task_requirements); assessments.append(assessment)
            if assessment.exact_recovery_required:
                mechanism=Mechanism.TOKENIZE.value; surrogate=self.vault.protect(entity.text,entity.entity_type,mechanism)
            else:
                mechanism=Mechanism.DP_NOISE.value
                try:
                    self.budget.reserve(self.epsilon,f"{entity.entity_type}:{entity.start}")
                    surrogate=exponential_sanitize(entity.text,entity.entity_type,self.epsilon) if self.use_exponential_mechanism else sanitize_entity(entity.text,entity.entity_type,self.dp_mechanism,self.epsilon)
                except PrivacyBudgetExceeded:
                    try:
                        surrogate=semantic_generalize(entity.text, entity.entity_type)
                    except ValueError:
                        surrogate="a related detail"
                except ValueError:
                    surrogate=sanitize_entity(entity.text,entity.entity_type,self.dp_mechanism,self.epsilon)
                self.vault.store_mapping(surrogate,entity.text,entity.entity_type,mechanism)
            protected=protected[:entity.start]+surrogate+protected[entity.end:]
            records.append(ProtectedEntity(entity.text,surrogate,entity.entity_type,mechanism,entity.start,entity.end,assessment))
        report=self.leakage_detector.check(protected,records); retries=0
        while not report.safe and retries<self.max_leakage_retries:
            retries+=1
            for record in records:
                if record.original in protected:
                    replacement=self.vault.protect(record.original,record.entity_type,Mechanism.TOKENIZE.value); protected=protected.replace(record.original,replacement)
            report=self.leakage_detector.check(protected,records)
        if not report.safe: raise ProtectionBlocked("protected prompt failed leakage checks")
        return ProtectionResult(text,protected,records,assessments,policy,report,metadata={"leakage_retries":retries,"budget":self.budget.summary()})
    def process(self,text,task_requirements=None,authorization=None):
        if self.model_router is None: raise ProviderUnavailable("no model router is configured")
        result=self.protect(text,task_requirements)
        utility_decision = None
        if self.utility_assessor is not None:
            features = extract_features(text, result.protected_text, self.epsilon,
                                        len(result.protected_entities), embed_fn=self.embed_fn)
            utility_decision = self.utility_assessor.assess(features)
            if not utility_decision.allowed and self.model_router.local_call is None:
                raise ProtectionBlocked("protected prompt failed utility threshold")
        assessment=max(result.assessments,key=lambda x:x.risk_score,default=self.risk_engine.assess("","UNKNOWN"))
        route=self.model_router.choose(assessment,result.policy.action); response=self.model_router.call(route,result.protected_text)
        filtered=self.response_filter.filter(response,result.protected_entities)
        if not filtered.allowed: raise ProtectionBlocked("model response failed privacy filtering")
        if authorization is not None: self.vault.authorize_restoration(authorization)
        return self.vault.restore_all(filtered.response)
