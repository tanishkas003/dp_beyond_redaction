"""Cloud/local provider policy abstraction."""
from dataclasses import dataclass
from src.risk_engine import RiskCategory
class ProviderUnavailable(RuntimeError): pass
@dataclass(frozen=True)
class ModelRoute: provider: str; reason: str
class ModelRouter:
    def __init__(self,cloud_call=None,local_call=None): self.cloud_call=cloud_call; self.local_call=local_call
    def choose(self,assessment,policy_action="CLOUD_OR_LOCAL"):
        if policy_action=="BLOCK": return ModelRoute("BLOCK","critical-data policy blocks processing")
        if policy_action=="LOCAL_ONLY" or assessment.category in {RiskCategory.HIGH,RiskCategory.CRITICAL}:
            return ModelRoute("LOCAL","risk policy requires local processing") if self.local_call else ModelRoute("BLOCK","no approved local provider is configured")
        if self.cloud_call: return ModelRoute("CLOUD","risk level permits cloud processing after checks")
        return ModelRoute("LOCAL","cloud provider unavailable") if self.local_call else ModelRoute("BLOCK","no approved provider is configured")
    def call(self,route,prompt):
        fn=self.cloud_call if route.provider=="CLOUD" else self.local_call if route.provider=="LOCAL" else None
        if fn: return fn(prompt)
        raise ProviderUnavailable(route.reason)
