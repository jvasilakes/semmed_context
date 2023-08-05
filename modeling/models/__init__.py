from .util import (MODEL_REGISTRY, TASK_ENCODER_REGISTRY, ENTITY_POOLER_REGISTRY, LOSS_REGISTRY)  # noqa F401

# The imports below populate LOSS_REGISTRY
from .losses import InverseFocalLoss  # noqa F401

# The imports below populate TASK_ENCODER_REGISTRY
from .task_encoders import BertAttentionEncoder  # noqa F401

# The imports below populate ENTITY_POOLER_REGISTRY
from .entity_poolers import (FirstEntityPooler, MaxEntityPooler,  # noqa F401
                             SoftmaxAttentionEntityPooler)  # noqa F401

# The imports below populate MODEL_REGISTRY
from .default import BertForMultiTaskSequenceClassification  # noqa F401
from .solid_markers import SolidMarkerClassificationModel  # noqa F401
from .levitated_markers import LevitatedMarkerClassificationModel  # noqa F401
from .levitated_markers_attentions import LevitatedMarkerModelWithAttentions  # noqa F401
from .levitated_markers_hier import LevitatedMarkerHierModel  # noqa F401
