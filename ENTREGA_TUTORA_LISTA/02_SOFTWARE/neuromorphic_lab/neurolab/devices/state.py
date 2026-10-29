from neurolab.devices.config import StateConfig

class StateManager:
    """
    Gestión del estado interno normalizado x in [x_min, x_max].
    """

    def __init__(self, config: StateConfig):
        self.config = config
        self._x: float = float(config.x_init)

    @property
    def value(self) -> float:
        return self._x

    @value.setter
    def value(self, val: float) -> None:
        v = float(val)
        self._x = self.config.x_max if v > self.config.x_max else (self.config.x_min if v < self.config.x_min else v)

    def reset(self, new_init: float = None) -> None:
        val = float(new_init if new_init is not None else self.config.x_init)
        self._x = self.config.x_max if val > self.config.x_max else (self.config.x_min if val < self.config.x_min else val)
