from dataclasses import dataclass
from typing import List,Optional

from src.detection import DetectedEntity, HybridDetector
from src.mechanism_router import MechanismRouter
from src.vault import SessionVault
from src.mechanism_router import MechanismRouter, Mechanism
from src.tokenization import Tokenizer
from src.dp_noise import sanitize_entity, MetricDPMechanism


@dataclass
class ProtectedEntity:
    original: str
    surrogate: str
    entity_type: str
    mechanism: str
    start: int
    end: int


@dataclass
class ProtectionResult:
    original_text: str
    protected_text: str
    protected_entities: List[ProtectedEntity]


class ProtectionPipeline:
    """
    End-to-end protection pipeline.

    Flow:

        Text
          ↓
        Detection
          ↓
        Routing
          ↓
        Vault protection
          ↓
        Protected text
    """

    def __init__(
    self,
    detector: HybridDetector,
    router: MechanismRouter,
    vault: SessionVault,
    tokenizer: Optional[Tokenizer] = None,
    dp_mechanism: Optional[MetricDPMechanism] = None,
    epsilon: float = 5.0,
    ):
        self.detector = detector
        self.router = router
        self.vault = vault

        self.tokenizer = tokenizer or Tokenizer()

        self.dp_mechanism = dp_mechanism
        self.epsilon = epsilon

    def protect(self, text: str) -> ProtectionResult:

        # --------------------------------------------------
        # STEP 1: Detect entities
        # --------------------------------------------------

        entities = self.detector.detect(text)

        # --------------------------------------------------
        # STEP 2: Sort entities from right to left
        # --------------------------------------------------
        #
        # Replacing text from the end prevents earlier
        # replacements from changing the character positions
        # of later entities.
        # --------------------------------------------------

        entities = sorted(
            entities,
            key=lambda entity: entity.start,
            reverse=True,
        )

        protected_text = text
        protected_entities = []

        # --------------------------------------------------
        # STEP 3: Route + protect every entity
        # --------------------------------------------------

        for entity in entities:

            # ----------------------------------------------
            # 1. Ask router which protection mechanism
            # ----------------------------------------------

            decision = self.router.route(
        entity.text,
        entity.entity_type,
        text,
            )

            # ----------------------------------------------
            # 2. TOKENIZATION BRANCH
            # ----------------------------------------------

            if decision.mechanism == Mechanism.TOKENIZE:

                surrogate = self.tokenizer.tokenize(
                    entity.entity_type
                )

            # ----------------------------------------------
            # 3. DP-NOISE BRANCH
            # ----------------------------------------------

            elif decision.mechanism == Mechanism.DP_NOISE:

                surrogate = sanitize_entity(
            entity_text=entity.text,
            entity_type=entity.entity_type,
            mechanism=self.dp_mechanism,
            epsilon=self.epsilon,
                )

            else:
                # Safety fallback
                surrogate = entity.text

            # ----------------------------------------------
            # 4. Store encrypted mapping in vault
            # ----------------------------------------------

            if surrogate != entity.text:

                self.vault.store_mapping(
            surrogate=surrogate,
            original=entity.text,
            entity_type=entity.entity_type,
            mechanism=decision.mechanism.value,
                )

            # Replace ONLY using detected character positions
            protected_text = (
                protected_text[:entity.start]
                + surrogate
                + protected_text[entity.end:]
            )

            protected_entities.append(
                ProtectedEntity(
                    original=entity.text,
                    surrogate=surrogate,
                    entity_type=entity.entity_type,
                    mechanism=decision.mechanism.value,
                    start=entity.start,
                    end=entity.end,
                )
            )

        return ProtectionResult(
            original_text=text,
            protected_text=protected_text,
            protected_entities=protected_entities,
        )