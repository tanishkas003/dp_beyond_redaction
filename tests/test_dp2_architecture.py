import pytest
from src.critical_policy import CriticalDataPolicy
from src.exponential_mechanism import ExponentialMechanism, ExponentialMechanismConfig
from src.privacy_budget import PrivacyBudgetAccountant, PrivacyBudgetExceeded
from src.risk_engine import PrivacyRiskEngine, RiskCategory
from src.vault import SessionVault
from src.generalization import generalize
from src.leakage_detector import LeakageDetector
from src.protection_pipeline import ProtectionPipeline, ProtectionBlocked
from src.detection import DetectedEntity
from src.mechanism_router import MechanismRouter
from src.model_router import ModelRouter
from src.utility_assessor import UtilityAssessor

def test_risk_assessment_is_explainable():
    result=PrivacyRiskEngine().assess("123-45-6789","SSN","preserve the exact value")
    assert result.category in {RiskCategory.HIGH,RiskCategory.CRITICAL}
    assert result.exact_recovery_required
    assert len(result.factors.as_dict()) == 6
def test_critical_policy_blocks(): assert not CriticalDataPolicy().inspect("password: secret").allowed
def test_exponential_probabilities_sum_to_one():
    m=ExponentialMechanism(ExponentialMechanismConfig())
    p=m.probabilities("x",["a","b"],lambda x,y: 1 if y=="a" else 0)
    assert sum(p.values()) == pytest.approx(1)
def test_exponential_rejects_empty_candidates():
    with pytest.raises(ValueError): ExponentialMechanism().probabilities("x",[],lambda x,y: 0)
def test_budget_exhaustion():
    b=PrivacyBudgetAccountant(1); b.reserve(.75)
    with pytest.raises(PrivacyBudgetExceeded): b.reserve(.5)
def test_generalization_categories():
    assert generalize("15 March 2026","DATE")=="March 2026"
    assert generalize("23 years old","AGE")=="20-29 years old"
def test_vault_authorization_and_tamper():
    v=SessionVault(require_authorization=True); token=v.protect("Priya Sharma","PERSON","TOKENIZE")
    with pytest.raises(PermissionError): v.restore(token)
    v.authorize_restoration("capability"); assert v.restore(token)=="Priya Sharma"
    v._encrypted_store[token]=b"tampered"
    with pytest.raises(ValueError): v.restore(token)

def test_vault_isolated_between_sessions_and_destroyed_sessions():
    first = SessionVault()
    token = first.protect("Priya Sharma", "PERSON", "TOKENIZE")
    second = SessionVault()
    assert second.restore(token) is None
    first.destroy()
    assert first.restore(token) is None

def test_critical_policy_uses_local_provider_without_cloud_fallback():
    text = "password: secret"
    entity = DetectedEntity(text, "PASSWORD", 0, len(text), "regex")
    pipeline = ProtectionPipeline(
        StaticDetector([entity]), MechanismRouter(), SessionVault(),
        critical_policy=CriticalDataPolicy(allow_local=True),
        model_router=ModelRouter(
            cloud_call=lambda prompt: (_ for _ in ()).throw(AssertionError("cloud used")),
            local_call=lambda prompt: "local: " + prompt,
        ),
    )
    assert "secret" in pipeline.process(text)
def test_leakage_detector(): assert not LeakageDetector().check("user@example.com").safe

class StaticDetector:
    def __init__(self, entities): self.entities = entities
    def detect(self, text): return self.entities

def test_utility_gate_blocks_low_utility_cloud_request():
    text = "The office is in Bangalore."
    start = text.index("Bangalore")
    entity = DetectedEntity("Bangalore", "LOCATION", start, start + 9, "regex")
    pipeline = ProtectionPipeline(
        StaticDetector([entity]), MechanismRouter(), SessionVault(),
        model_router=ModelRouter(cloud_call=lambda prompt: prompt),
        utility_assessor=UtilityAssessor(quality_threshold=1.0),
        use_exponential_mechanism=False,
    )
    with pytest.raises(ProtectionBlocked):
        pipeline.process(text)

def test_budget_exhaustion_falls_back_without_spending_more_epsilon():
    text = "The office is in Bangalore."
    start = text.index("Bangalore")
    entity = DetectedEntity("Bangalore", "LOCATION", start, start + 9, "regex")
    budget = PrivacyBudgetAccountant(total_epsilon=0.5)
    result = ProtectionPipeline(
        StaticDetector([entity]), MechanismRouter(), SessionVault(),
        budget=budget, epsilon=1.0,
    ).protect(text)
    assert result.protected_text != text
    assert budget.spent == 0
