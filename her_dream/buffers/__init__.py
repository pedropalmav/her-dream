from .buffer import Buffer
from .her_buffer import HERBuffer


def make_buffer(config, reward_function):
    match config.buffer.type:
        case "her":
            return HERBuffer(config.buffer, reward_function)
        case "normal":
            return Buffer(config.buffer)
        case _:
            raise ValueError(f"Tipo de buffer no soportado: {config.buffer.type}")
