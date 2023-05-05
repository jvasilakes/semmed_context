from .util import MODEL_REGISTRY, ENTITY_POOLER_REGISTRY

# The imports below populate MODEL_REGISTRY
from .default import BertForMultiTaskSequenceClassification
from .solid_markers import SolidMarkerClassificationModel

# The imports below populate ENTITY_POOLER_REGISTRY
from .entity_poolers import MaxEntityPooler
