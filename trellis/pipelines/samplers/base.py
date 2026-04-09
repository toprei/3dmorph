from typing import *
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.
        """
        pass
    
    @abstractmethod
    def inpaint(
        self,
        model,
        **kwargs
    ):
        """
        Inpaint using a model.
        """
        pass
    