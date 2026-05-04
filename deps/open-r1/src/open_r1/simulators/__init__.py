"""
Simulator interfaces for multi-domain engineering problems.
Provides clean, agentic-friendly access to simulators.
"""

from .base import BaseSimulator
from .eps import EPSSimulator
from .beams2d import Beams2DSimulator
from .knapsack import KnapsackSimulator
from .sds_simulator import SDSSimulator
from .decadal_simulator import DecadalSimulator
from .registry import SimulatorRegistry

__all__ = [
    'BaseSimulator',
    'EPSSimulator', 
    'Beams2DSimulator',
    'KnapsackSimulator',
    'SDSSimulator',
    'DecadalSimulator',
    'SimulatorRegistry'
]
