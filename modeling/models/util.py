MODEL_REGISTRY = {}


def register_model(name):
    def add_to_registry(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return add_to_registry


TASK_ENCODER_REGISTRY = {}


def register_task_encoder(name):
    def add_to_registry(cls):
        TASK_ENCODER_REGISTRY[name] = cls
        return cls
    return add_to_registry


ENTITY_POOLER_REGISTRY = {}

def register_entity_pooler(name):
    def add_to_registry(cls):
        ENTITY_POOLER_REGISTRY[name] = cls
        return cls
    return add_to_registry


LOSS_REGISTRY = {}


def register_loss(name):
    def add_to_registry(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return add_to_registry
