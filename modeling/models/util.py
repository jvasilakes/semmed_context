MODEL_REGISTRY = {}


def register_model(name):
    def add_to_registry(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return add_to_registry


ENTITY_POOLER_REGISTRY = {}


def register_entity_pooler(name):
    def add_to_registry(cls):
        ENTITY_POOLER_REGISTRY[name] = cls
        return cls
    return add_to_registry
