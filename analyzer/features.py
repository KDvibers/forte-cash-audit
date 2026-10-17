"""Shared vocabulary of visible state observations, not actions or verdicts.

The optional cash features describe visible banknotes on the work surface, in
hands, or in a machine. They do not establish ownership, authenticity, counting,
or the absence of an action between sampled frames. A missing field means it
was not observed; it must never be supplied with an implicit ``no`` default.
"""

from types import MappingProxyType


BINARY_STATES = ("yes", "no", "uncertain")
CASH_FEATURES = ("cash_on_worktop", "cash_in_hands", "notes_in_machine")
UNKNOWN_STATES = frozenset({"uncertain", "not_visible"})

# Preserve the order and values of the existing person/cabinet features.
FEATURE_STATES = MappingProxyType({
    "person_in_work_area": BINARY_STATES,
    "cabinet_door": ("open", "closed", "uncertain", "not_visible"),
    **{feature: BINARY_STATES for feature in CASH_FEATURES},
})
KNOWN_STATES = MappingProxyType({
    feature: frozenset(states) - UNKNOWN_STATES
    for feature, states in FEATURE_STATES.items()
})
