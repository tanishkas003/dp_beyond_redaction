"""Fail-closed policy for secrets and credentials."""
import re
from dataclasses import dataclass
@dataclass(frozen=True)
class PolicyDecision: allowed: bool; action: str; reason: str
class CriticalDataPolicy:
    PATTERNS=(("PASSWORD",re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*\S+")),("PRIVATE_KEY",re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),("API_KEY",re.compile(r"(?i)\b(?:api[_ -]?key|secret[_ -]?key)\s*[:=]\s*\S+")),("AUTH_TOKEN",re.compile(r"(?i)\b(?:bearer|access[_ -]?token|auth[_ -]?token)\s*[:=]\s*\S+")))
    def __init__(self,allow_local=False,block_types=None): self.allow_local=allow_local; self.block_types={x.upper() for x in (block_types or set())}
    def inspect(self,text,entity_types=None):
        types={x.upper() for x in (entity_types or set())}
        for kind,pattern in self.PATTERNS:
            if pattern.search(text) or kind in types or kind in self.block_types:
                return PolicyDecision(True,"LOCAL_ONLY","critical data requires a trusted local provider") if self.allow_local else PolicyDecision(False,"BLOCK","critical data is not approved for cloud processing")
        return PolicyDecision(True,"CLOUD_OR_LOCAL","no critical policy match")
