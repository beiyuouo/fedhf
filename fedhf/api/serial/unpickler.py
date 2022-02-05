import io
import pickle
import numpy as np
import torch

from ..message import Message


class Unpickler(pickle.Unpickler):
    __SAFE_CLASSES = {
        "torch.Tensor": torch.Tensor,
        "fedhf.api.message": Message,
    }

    @classmethod
    def register_safe_class(cls, input_class):
        assert isinstance(input_class,
                          type), "Cannot register %s type as safe" % type(input_class)
        classname = str(input_class).split("'")[1]
        cls.__SAFE_CLASSES[classname] = input_class

    def find_class(self, module, name):
        classname = f'{module}.{name}'
        if classname in self.__SAFE_CLASSES.keys():
            return self.__SAFE_CLASSES[classname]
        else:
            raise pickle.UnpicklingError(f"{classname} is not a safe class to unpickle")
