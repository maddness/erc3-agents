"""
Evolution module for self-improving agent.

Components:
- Versioner: управление версиями config
- Analyzer: анализ провалов с Haiku
- Evolver: генерация улучшений с Haiku
- EvolutionRunner: оркестратор цикла эволюции
"""

from .versioner import Versioner
from .analyzer import Analyzer, AnalysisResult
from .evolver import Evolver, EvolutionResult
from .runner import EvolutionRunner

__all__ = [
    "Versioner",
    "Analyzer",
    "AnalysisResult",
    "Evolver",
    "EvolutionResult",
    "EvolutionRunner"
]
