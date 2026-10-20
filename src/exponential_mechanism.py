"""Numerically stable Exponential Mechanism implementation."""
import math, random
from dataclasses import dataclass
@dataclass(frozen=True)
class ExponentialMechanismConfig:
    epsilon: float = 1.0; sensitivity: float = 1.0
    def __post_init__(self):
        if self.epsilon <= 0 or self.sensitivity <= 0: raise ValueError("epsilon and sensitivity must be positive")
class ExponentialMechanism:
    def __init__(self, config=None, rng=None): self.config=config or ExponentialMechanismConfig(); self.rng=rng or random.Random()
    def probabilities(self, original, candidates, utility):
        choices=list(dict.fromkeys(candidates))
        if not choices: raise ValueError("candidate set must not be empty")
        scale=self.config.epsilon/(2*self.config.sensitivity); scores=[scale*float(utility(original,c)) for c in choices]; top=max(scores)
        weights=[math.exp(x-top) for x in scores]; total=sum(weights)
        return {c:w/total for c,w in zip(choices,weights)}
    def select(self, original, candidates, utility):
        probs=self.probabilities(original,candidates,utility); value=self.rng.random(); total=0
        for candidate, probability in probs.items():
            total += probability
            if value <= total: return candidate
        return next(reversed(probs))
