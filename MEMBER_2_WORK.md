# Member 2 Work Handoff

This document records the work completed by Member 2 for the Beyond Redaction project. It is intended as a reference for the other members when integrating and extending the project.

## Working Branch

The work was completed on:

`member2-privacy-vault`

## Completed Work

### Encrypted Session Vault

Implemented and verified in `src/vault.py`:

- Added `SessionVault` for session-scoped reversible mappings.
- Stores mappings for both `TOKENIZE` and `DP_NOISE` mechanisms.
- Encrypts every mapping with Fernet before placing it in the in-memory store.
- Keeps the Fernet key in memory for the active session.
- Restores one value with `restore()`.
- Restores all known protected values in an LLM response with `restore_all()`.
- Clears mappings and replaces the session key when `destroy()` is called.
- Generates typed placeholders for entity types without a semantic surrogate pool.
- Uses one-to-many surrogate pools for supported semantic entity types.
- Normalizes entity types and mechanism names to uppercase, so lowercase inputs work consistently.

Example usage:

```python
vault = SessionVault()
protected_value = vault.protect(
    "Priya Sharma",
    "PERSON",
    mechanism="TOKENIZE",
)
original_value = vault.restore(protected_value)
final_response = vault.restore_all(protected_llm_response)
vault.destroy()
```

### DP-Noise Branch

Implemented and verified in `src/dp_noise.py`:

- Added category-based generalization fallback for `LOCATION`, `JOB_TITLE`, `DATE`, and `ORG`.
- Added a generic fallback for unknown entity types.
- Made entity type lookup case-insensitive.
- Added `MetricDPMechanism` for optional embedding-space noise and nearest-vocabulary decoding.
- Added `MetricDPConfig` with configurable epsilon and embedding dimension.
- Added `sanitize_entity()` to select fallback generalization when no metric-DP mechanism is configured.

Fallback examples:

| Entity type | Protected value |
| --- | --- |
| `LOCATION` | `a regional office` |
| `JOB_TITLE` | `a professional role` |
| `DATE` | `a recent date` |
| `ORG` | `a partner organization` |
| Unknown type | `a related detail` |

### Tests Added and Verified

Updated:

- `tests/test_vault.py`
- `tests/test_dp_noise.py`

The tests cover:

- Tokenized entity protection and restoration
- DP entity protection and restoration
- Restoration of multiple values in an LLM response
- Vault destruction and session cleanup
- Unique structured-entity placeholders
- Pre-generated mapping storage
- Encryption of mappings at rest
- Case-insensitive entity and mechanism names
- DP generalization for supported entity types
- Unknown-type generalization fallback
- DP fallback behavior when no metric mechanism is configured

Scoped test command:

```text
python -m pytest -q tests/test_vault.py tests/test_dp_noise.py
```

Result at handoff: `16 passed`

## Integration Contract For Other Members

The expected protected-response flow is:

```text
Original prompt
    -> ProtectionPipeline.protect()
    -> Protected prompt
    -> LLM/API
    -> Protected LLM response
    -> SessionVault.restore_all()
    -> Final response
```

The vault does not decide which privacy mechanism to use. The mechanism router or pipeline should provide the selected mechanism and protected value, then use `store_mapping()` when the protected value was generated outside the vault.

For example, the DP branch can be called as follows:

```python
protected_value = sanitize_entity(
    entity_text,
    entity_type,
    mechanism=None,
    epsilon=5.0,
)
vault.store_mapping(
    surrogate=protected_value,
    original=entity_text,
    entity_type=entity_type,
    mechanism="DP_NOISE",
)
```

## Work Not Completed In This Scope

The following items remain for the relevant project members:

- Connecting `ProtectionPipeline.protect()` to a real LLM or API client.
- Sending the protected prompt to the selected LLM/API.
- Passing the protected LLM response through `SessionVault.restore_all()` in the application flow.
- Measuring response utility, quality, privacy, and restoration success.
- Configuring and evaluating `MetricDPMechanism` with a real embedding model and domain vocabulary.
- Completing the full repository test suite, including optional GLiNER-dependent tests that may require model setup and additional runtime.

These items were not implemented in the Member 2 vault/DP-noise files.

## Files Owned By This Work

- `src/vault.py`
- `src/dp_noise.py`
- `tests/test_vault.py`
- `tests/test_dp_noise.py`

No files outside this Member 2 scope were modified for the implementation described here.
