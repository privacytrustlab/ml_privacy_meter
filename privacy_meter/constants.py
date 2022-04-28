from enum import Enum


class MetricEnum(Enum):
    POPULATION = "population_metric"
    SHADOW = "shadow_metric"
    REFERENCE = "reference_metric"


class SignalSourceEnum(Enum):
    TARGET_MEMBER = "target_member"
    TARGET_NON_MEMBER = "target_non_member"
    REFERENCE = "reference"
    REFERENCE_MEMBER = "reference_member"
    REFERENCE_NON_MEMBER = "reference_non_member"


class InferenceGame(Enum):
    AVG_PRIVACY_LOSS_TRAINING_ALGO = "Average privacy loss of a training algorithm"
    PRIVACY_LOSS_MODEL = "Privacy loss of a model"
    PRIVACY_LOSS_SAMPLE = "Privacy loss of a data record"
    WORST_CASE_PRIVACY_LOSS_TRAINING_ALGO = "Worst-case privacy loss of a training algorithm"


TARGET_MEMBER_SIGNALS_FILENAME = 'target_member_signals'
TARGET_NON_MEMBER_SIGNALS_FILENAME = 'target_non_member_signals'

REFERENCE_MEMBER_SIGNALS_FILENAME = 'reference_member_signals'
REFERENCE_NON_MEMBER_SIGNALS_FILENAME = 'reference_non_member_signals'
REFERENCE_SIGNALS_FILENAME = 'reference_signals'

NPZ_EXTENSION = '.npz'
