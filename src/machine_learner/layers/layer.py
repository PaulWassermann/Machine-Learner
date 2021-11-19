from abc import ABC, abstractmethod


class Layer(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def propagate_forward(self, x):
        ...

    @abstractmethod
    def propagate_backwards(self, x):
        ...
