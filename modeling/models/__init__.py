from .util import MODEL_REGISTRY, ENTITY_POOLER_REGISTRY  # noqa F401

# The imports below populate MODEL_REGISTRY
from .default import BertForMultiTaskSequenceClassification  # noqa F401
from .solid_markers import SolidMarkerClassificationModel  # noqa F401
from .levitated_markers import LevitatedMarkerClassificationModel  # noqa F401

# The imports below populate ENTITY_POOLER_REGISTRY
from .entity_poolers import MaxEntityPooler  # noqa F401
