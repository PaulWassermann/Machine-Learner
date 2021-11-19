import numpy.typing as npt
from typing import List, Dict, Union, Type

Layer = Type["Layer"]
npNumber = npt.DTypeLike
Number = Union[int, float]
NeuralNetwork = Type["NeuralNetwork"]
Architecture = Dict[str, Union[List[Dict[str, str]], List[str]]]
