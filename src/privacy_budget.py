"""Thread-safe basic composition: epsilon_total = sum(epsilon_i)."""
import threading
from dataclasses import dataclass
@dataclass(frozen=True)
class BudgetEvent: operation_id: str; epsilon: float; scope: str
class PrivacyBudgetExceeded(RuntimeError): pass
class PrivacyBudgetAccountant:
    def __init__(self,total_epsilon=10.0,scope="session"):
        if total_epsilon<=0 or not scope: raise ValueError("invalid privacy budget")
        self.total_epsilon=float(total_epsilon); self.scope=scope; self._spent=0.; self._events=[]; self._lock=threading.Lock()
    @property
    def spent(self):
        with self._lock: return self._spent
    @property
    def remaining(self): return max(0.,self.total_epsilon-self.spent)
    def reserve(self,epsilon,operation_id="dp-operation"):
        if not isinstance(epsilon,(int,float)) or epsilon<=0: raise ValueError("epsilon must be positive")
        with self._lock:
            if self._spent+epsilon>self.total_epsilon+1e-12: raise PrivacyBudgetExceeded("privacy budget exhausted")
            event=BudgetEvent(operation_id,float(epsilon),self.scope); self._events.append(event); self._spent+=epsilon; return event
    def summary(self): return {"scope":self.scope,"total_epsilon":self.total_epsilon,"spent_epsilon":self.spent,"remaining_epsilon":self.remaining,"operations":len(self._events)}
