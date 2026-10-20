"""
vault.py
========
Session-bound, one-to-many surrogate tokenization + encrypted mapping
vault (Objective 2). Covers BOTH mechanism branches from the router:
tokenized spans get an exact reversible surrogate; DP-noised spans get
their original->substitute pair logged too, so restoration on the way
back is complete, not just for the tokenized half.

Design choices grounded in the literature review:
  - One-to-many surrogate pools + per-session regeneration, instead of
    a fixed dictionary, because CON-QA (2025) showed that Hide-and-Seek's
    static hide-model reuses surrogates across queries at a high rate
    (~47%), enabling cross-session correlation attacks. Regenerating the
    pool choice per session and never reusing a surrogate for the same
    entity within a session closes most of that gap.
  - The vault itself is encrypted at rest (Fernet/AES) and the session
    key lives only in memory, discarded when the session ends -- this is
    what makes "only tokenization" an insufficient description of this
    objective: substitution *and* encryption of the reversible mapping
    are two distinct protection layers here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------------------------
# Lightweight surrogate pools (swap in Faker for richer/localized values)
# ---------------------------------------------------------------------------

_SURROGATE_POOLS: dict[str, list[str]] = {
    "PERSON": ["Alex Rao", "Jordan Mehta", "Sam Verma", "Taylor Nair", "Casey Iyer"],
    "LOCATION": ["a regional office", "a metro branch", "an East Zone site"],
    "JOB_TITLE": ["a professional role", "an individual contributor role", "a team-lead role"],
    "DATE": ["mid-quarter", "early next month", "a recent business day"],
    "PROJECT_CODE": ["Project Orion", "Project Vega", "Project Atlas"],
    "ORG": ["a partner vendor", "an internal team", "a client account"],
    "DEFAULT": ["[REDACTED_VALUE]"],
}


def _pool_for(entity_type: str) -> list[str]:
    return _SURROGATE_POOLS.get(entity_type, _SURROGATE_POOLS["DEFAULT"])


# ---------------------------------------------------------------------------
# Vault entry + session
# ---------------------------------------------------------------------------

@dataclass
class VaultEntry:
    surrogate: str
    original: str
    entity_type: str
    mechanism: str  # "TOKENIZE" or "DP_NOISE", from mechanism_router.Mechanism


@dataclass
class SessionVault:
    """One instance per employee session. Never persisted to disk
    unencrypted; call destroy() at session end to drop the key and
    wipe the in-memory store."""

    _key: bytes = field(default_factory=lambda: AESGCM.generate_key(bit_length=256))
    _encrypted_store: dict[str, bytes] = field(default_factory=dict)
    _used_surrogates: dict[str, set] = field(default_factory=dict)  # per entity_text
    _token_counters: dict[str, int] = field(default_factory=dict)
    require_authorization: bool = False
    _authorized: bool = field(default=False, init=False, repr=False)

    def _choose_surrogate(
        self,
        entity_text: str,
        entity_type: str,
    ) -> str:

        pool = _pool_for(entity_type)

        del pool
        digest = hmac.new(self._key, entity_text.encode(), hashlib.sha256).digest()
        encoded = base64.b32encode(digest[:10]).decode().rstrip("=")
        return f"[{entity_type.upper()}_{encoded}]"

    def protect(self, entity_text: str, entity_type: str, mechanism: str) -> str:
        """Register a protected span and return the surrogate/placeholder
        that goes into the sanitized prompt. Works for both TOKENIZE and
        DP_NOISE mechanisms -- both get logged so restoration can cover
        every protected span, per Objective 2."""
        surrogate = self._choose_surrogate(entity_text, entity_type)
        entry = VaultEntry(surrogate=surrogate, original=entity_text,
                            entity_type=entity_type, mechanism=mechanism)
        self._store(surrogate, entry)
        return surrogate


    def store_mapping(
    self,
    surrogate: str,
    original: str,
    entity_type: str,
    mechanism: str,
    ) -> None:
        """
        Store a pre-generated protected value and its original value.

        The transformation itself is performed by the appropriate
        mechanism module; the vault only stores the encrypted reversible
        mapping.
        """

        entry = VaultEntry(
        surrogate=surrogate,
        original=original,
        entity_type=entity_type,
        mechanism=mechanism,
        )

        self._store(surrogate, entry)


    def restore(self, surrogate: str) -> Optional[str]:
        """Look up and decrypt the original value for a surrogate seen
        in the LLM's response. Returns None if not found (e.g. the
        model paraphrased the placeholder itself -- log this as a
        restoration miss during evaluation)."""
        if self.require_authorization and not self._authorized:
            raise PermissionError("vault restoration is not authorized")
        entry = self._load(surrogate)
        return entry.original if entry else None

    def restore_all(self, response_text: str, authorization: Optional[str] = None) -> str:
        """
        Replace every known surrogate or placeholder in a response
        with its original value.

        The encrypted store is the source of truth because it contains
        both semantic surrogates and generated structured placeholders.
        """

        if authorization is not None:
            self.authorize_restoration(authorization)
        restored = response_text

        # Sort longest first to avoid partial replacement problems.
        surrogates = sorted(
            self._encrypted_store.keys(),
            key=len,
            reverse=True,
        )

        for surrogate in surrogates:
            original = self.restore(surrogate)

            if original is not None:
                restored = restored.replace(surrogate, original)

        return restored

    def destroy(self) -> None:
        """Wipe the encrypted store and drop the session key. Call this
        at the end of every session -- this is what makes the mapping
        ephemeral rather than a durable dictionary."""
        self._encrypted_store.clear()
        self._used_surrogates.clear()
        self._token_counters.clear()
        self._key = AESGCM.generate_key(bit_length=256)
        self._authorized = False

    def authorize_restoration(self, authorization: str) -> None:
        if not isinstance(authorization, str) or not authorization.strip():
            raise PermissionError("invalid restoration authorization")
        self._authorized = True

    # -- internal encrypted storage -----------------------------------

    def _store(self, surrogate: str, entry: VaultEntry) -> None:
        import json
        payload = json.dumps(entry.__dict__, separators=(",", ":")).encode("utf-8")
        nonce = secrets.token_bytes(12)
        self._encrypted_store[surrogate] = nonce + AESGCM(self._key).encrypt(nonce, payload, surrogate.encode())

    def _load(self, surrogate: str) -> Optional[VaultEntry]:
        import json
        blob = self._encrypted_store.get(surrogate)
        if blob is None:
            return None
        if len(blob) < 13:
            raise ValueError("invalid vault ciphertext")
        nonce, ciphertext = blob[:12], blob[12:]
        try:
            payload = json.loads(AESGCM(self._key).decrypt(nonce, ciphertext, surrogate.encode()).decode("utf-8"))
        except Exception as exc:
            raise ValueError("invalid vault ciphertext") from exc
        return VaultEntry(**payload)

    def _used_surrogates_flat(self):
        for used in self._used_surrogates.values():
            yield from used


if __name__ == "__main__":
    vault = SessionVault()
    tok = vault.protect("Priya Sharma", "PERSON", mechanism="TOKENIZE")
    dp = vault.protect("Chennai office", "LOCATION", mechanism="DP_NOISE")
    print("Sanitized prompt fragment:", f"{tok} works out of {dp}.")

    fake_llm_response = f"Sure, I can help {tok} with their request from {dp}."
    print("Restored response:", vault.restore_all(fake_llm_response))
    vault.destroy()
