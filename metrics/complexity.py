# metrics/complexity.py
from .entropy import entropy
from .luca import luca_distance

def complexity(p):
    return entropy(p) + luca_distance(p)