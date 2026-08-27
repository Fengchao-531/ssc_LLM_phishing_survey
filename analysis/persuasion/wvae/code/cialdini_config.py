RAW_STRATEGY_NAMES = {
    0: "other",
    1: "credibility",
    2: "reciprocity",
    3: "evidence",
    4: "commitment",
    5: "scarcity",
    6: "social_identity",
    7: "emotion",
    8: "impact",
    9: "politeness",
}

PRINCIPLE_ORDER = [
    "authority",
    "reciprocity",
    "commitment",
    "scarcity",
    "social_proof",
    "liking",
]

# The WVAE paper uses request-specific rhetorical strategy labels rather than
# native Cialdini principles, so we aggregate the sentence-level strategy
# probabilities into the six principles we want to visualize downstream.
PRINCIPLE_STRATEGY_MAP = {
    "authority": [1, 3],
    "reciprocity": [2],
    "commitment": [4],
    "scarcity": [5],
    "social_proof": [6, 8],
    "liking": [7, 9],
}

IDENTITY_LABEL_MAPPING = {label_id: label_id for label_id in RAW_STRATEGY_NAMES}

PRINCIPLE_NOTES = {
    "authority": "Credibility and Evidence act as the closest authority-style proxy.",
    "reciprocity": "Directly mapped from Reciprocity.",
    "commitment": "Directly mapped from Commitment.",
    "scarcity": "Directly mapped from Scarcity.",
    "social_proof": "Social Identity and Impact are used as the closest social-proof proxy.",
    "liking": "Emotion and Politeness act as the closest liking proxy.",
}
