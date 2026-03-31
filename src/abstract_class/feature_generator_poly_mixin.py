import itertools
import math
from typing import List

import numpy as np


class FeatureGeneratorPolyMixin:
    @staticmethod
    def sniper_features(
        x: np.ndarray,
        degree: List[int],
        derivative: List[int] = None,
    ) -> np.ndarray:
        """Generate polynomial features for arbitrary dimensions."""
        x = np.asarray(x, dtype=np.float64)
        degree, derivative = FeatureGeneratorPolyMixin._normalize_poly_inputs(
            x.shape[1],
            degree,
            derivative,
        )
        powers = [range(d + 1) for d in degree]
        if np.all(derivative == 0):
            return FeatureGeneratorPolyMixin._build_polynomial_terms(x, powers)
        return FeatureGeneratorPolyMixin._build_derivative_terms(x, powers, derivative)

    @staticmethod
    def _normalize_poly_inputs(n_dims: int, degree, derivative):
        if isinstance(degree, int):
            degree = [degree] * n_dims
        if derivative is None:
            derivative = [0] * n_dims
        if isinstance(derivative, int):
            derivative = [derivative] * n_dims

        degree = np.asarray(degree, dtype=int)
        derivative = np.asarray(derivative, dtype=int)
        if degree.size != n_dims:
            degree = np.resize(degree, n_dims)
        if derivative.size != n_dims:
            derivative = np.resize(derivative, n_dims)
        if np.any(degree < 0) or np.any(derivative < 0):
            raise ValueError("degrees and derivative orders cannot be negative")
        if np.any(derivative > degree):
            raise ValueError("derivative order cannot exceed polynomial degree")
        return degree, derivative

    @staticmethod
    def _build_polynomial_terms(x: np.ndarray, powers) -> np.ndarray:
        terms = []
        for exponents in itertools.product(*powers):
            term = np.ones((x.shape[0], 1))
            for dim, power in enumerate(exponents):
                if power > 0:
                    term *= x[:, dim : dim + 1] ** power
            terms.append(term)
        return np.hstack(terms)

    @staticmethod
    def _build_derivative_terms(
        x: np.ndarray,
        powers,
        derivative: np.ndarray,
    ) -> np.ndarray:
        terms = []
        for exponents in itertools.product(*powers):
            terms.append(
                FeatureGeneratorPolyMixin._build_derivative_term(
                    x,
                    exponents,
                    derivative,
                )
            )
        return np.hstack(terms)

    @staticmethod
    def _build_derivative_term(
        x: np.ndarray,
        exponents,
        derivative: np.ndarray,
    ) -> np.ndarray:
        if not all(exp >= der for exp, der in zip(exponents, derivative)):
            return np.zeros((x.shape[0], 1))

        coefficient = 1.0
        term = np.ones((x.shape[0], 1))
        for dim, (exp, der) in enumerate(zip(exponents, derivative)):
            if der > 0:
                coefficient *= math.factorial(exp) / math.factorial(exp - der)
            if exp - der > 0:
                term *= x[:, dim : dim + 1] ** (exp - der)
        return term * coefficient
