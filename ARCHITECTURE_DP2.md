# DP 2.0 Architecture

The request flow is: detection -> per-entity risk assessment -> critical-data
policy -> mechanism selection and epsilon accounting -> cryptographic
pseudonymization, Exponential Mechanism, semantic generalization, or block/local
processing -> leakage check -> approved provider -> response privacy filter ->
authorized vault restoration.

HMAC-SHA256 pseudonyms are cryptographic pseudonyms, not differential privacy.
The Exponential Mechanism is implemented with a documented utility function and
sensitivity parameter, but a formal end-to-end guarantee still requires the
deployment's neighboring-input definition, candidate-set assumptions, and
composition analysis.
