from enum import Enum


class MetricEnum(Enum):
    POPULATION = "population"
    SHADOW = "shadow"


class SignalSourceEnum(Enum):
    TARGET_MEMBER = "target_member"
    TARGET_NON_MEMBER = "target_non_member"
    REFERENCE = "reference"
    REFERENCE_MEMBER = "reference_member"
    REFERENCE_NON_MEMBER = "reference_non_member"


TARGET_MEMBER_SIGNALS_FILENAME = 'target_member_signals'
TARGET_NON_MEMBER_SIGNALS_FILENAME = 'target_non_member_signals'

REFERENCE_MEMBER_SIGNALS_FILENAME = 'reference_member_signals'
REFERENCE_NON_MEMBER_SIGNALS_FILENAME = 'reference_non_member_signals'
REFERENCE_SIGNALS_FILENAME = 'reference_signals'

NPZ_EXTENSION = '.npz'
