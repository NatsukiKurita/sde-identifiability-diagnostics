KINETICS_REGISTRY       = {}
SOCIAL_REGISTRY         = {}

def register_kinetics(name):
    def wrapper(func):
        KINETICS_REGISTRY[name] = func
        return func
    return wrapper

def register_social(name):
    def wrapper(func):
        SOCIAL_REGISTRY[name] = func
        return func
    return wrapper

