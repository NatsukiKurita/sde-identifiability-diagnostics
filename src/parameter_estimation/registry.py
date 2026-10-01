KINETICS_REGISTRY       = {}
SOCIAL_REGISTRY         = {}
WALL_AVOIDANCE_REGISTRY = {}
WALL_AVOIDANCE_CIRCLE_REGISTRY = {}
NEIGHBOR_REGISTRY       = {}

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

def register_wall_avoidance(name):
    def wrapper(func):
        WALL_AVOIDANCE_REGISTRY[name] = func
        return func
    return wrapper


def register_wall_avoidance_circle(name):
    def wrapper(func):
        WALL_AVOIDANCE_CIRCLE_REGISTRY[name] = func
        return func
    return wrapper


def get_wall_avoidance_stepwise(wa_type: str, boundary_shape: str):
    if boundary_shape == "CircleWall":
        if wa_type not in WALL_AVOIDANCE_CIRCLE_REGISTRY:
            raise ValueError(
                f"CircleWall stepwise estimator is not implemented for {wa_type}"
            )
        return WALL_AVOIDANCE_CIRCLE_REGISTRY[wa_type]
    if wa_type not in WALL_AVOIDANCE_REGISTRY:
        raise KeyError(f"Unknown wall avoidance model: {wa_type}")
    return WALL_AVOIDANCE_REGISTRY[wa_type]

def register_neighbor(name):
    def wrapper(func):
        NEIGHBOR_REGISTRY[name] = func
        return func
    return wrapper

