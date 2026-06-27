"""Top-level package for safety_evaluation.

This file re-exports a small set of convenience functions that older
scripts and notebooks import directly from
``src.safety_evaluation`` (for example ``get_gamma``).

Keeping lightweight re-exports here avoids changing many import sites
in the codebase.
"""

def get_gamma(m_upper_bound: float, budget_per_sample: float):
	"""Return gamma = m_upper_bound / budget_per_sample.

	Imported lazily to avoid circular imports when other submodules
	import from ``src.safety_evaluation`` during package initialization.
	"""
	from .calibration.survival_calibration_with_known_weights import get_gamma as _get_gamma
	return _get_gamma(m_upper_bound, budget_per_sample)


__all__ = [
	'get_gamma',
]


