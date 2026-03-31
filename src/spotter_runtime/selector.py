from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class SpotterCandidateReport:
    name: str
    backend: str
    runtime: str
    score: float
    metric_used: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpotterSelectionReport:
    mode: str
    selected_name: str
    selected_backend: str
    metric: str
    candidates: list[SpotterCandidateReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_name": self.selected_name,
            "selected_backend": self.selected_backend,
            "metric": self.metric,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def score_provider_candidate(
    *,
    provider,
    candidate,
    selector_cfg: Dict[str, Any],
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
    reference_solution: Optional[Dict[str, Any]] = None,
) -> SpotterCandidateReport:
    metric = str(selector_cfg.get("metric", "auto") or "auto").lower()
    max_points = int(selector_cfg.get("max_points", 2048))
    priority_score = float(getattr(candidate, "priority", 0))
    weights = dict(selector_cfg.get("weights", {}) or {})
    boundary_score, boundary_metrics = _boundary_residual(provider, data_train)
    dataset_points = None
    target_values = None
    metric_used = "priority"
    rel_l2 = None
    abs_l2 = None

    if metric in {"auto", "test_l2"}:
        dataset_points, target_values = _dataset_from_data_test(data_test, max_points=max_points)
        if dataset_points is not None:
            metric_used = "test_l2"
    if dataset_points is None and metric in {"auto", "reference_l2", "test_l2", "composite"}:
        dataset_points, target_values = _dataset_from_reference(reference_solution, max_points=max_points)
        if dataset_points is not None:
            metric_used = "reference_l2"

    if dataset_points is not None and target_values is not None:
        prediction = np.asarray(provider.predict_output(dataset_points), dtype=np.float64)
        target = np.asarray(target_values, dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        n_cols = min(prediction.shape[1], target.shape[1])
        prediction = prediction[:, :n_cols]
        target = target[:, :n_cols]
        abs_l2 = float(np.linalg.norm(prediction - target))
        rel_l2 = abs_l2 / (float(np.linalg.norm(target)) + 1e-12)

    if metric == "boundary_residual":
        return SpotterCandidateReport(
            name=candidate.name,
            backend=candidate.backend,
            runtime=candidate.runtime,
            score=boundary_score if boundary_score is not None else priority_score,
            metric_used="boundary_residual" if boundary_score is not None else "priority",
            metrics=boundary_metrics if boundary_score is not None else {"priority": priority_score},
        )

    if metric == "composite":
        score, metrics = _composite_score(
            relative_l2=rel_l2,
            absolute_l2=abs_l2,
            boundary_score=boundary_score,
            boundary_metrics=boundary_metrics,
            priority_score=priority_score,
            weights=weights,
        )
        return SpotterCandidateReport(
            name=candidate.name,
            backend=candidate.backend,
            runtime=candidate.runtime,
            score=score,
            metric_used="composite",
            metrics=metrics,
        )

    if rel_l2 is None or abs_l2 is None:
        metrics = {"priority": priority_score}
        if boundary_score is not None:
            metrics.update(boundary_metrics)
        return SpotterCandidateReport(
            name=candidate.name,
            backend=candidate.backend,
            runtime=candidate.runtime,
            score=priority_score,
            metric_used=metric_used,
            metrics=metrics,
        )
    return SpotterCandidateReport(
        name=candidate.name,
        backend=candidate.backend,
        runtime=candidate.runtime,
        score=rel_l2,
        metric_used=metric_used,
        metrics={
            "relative_l2": rel_l2,
            "absolute_l2": abs_l2,
            "samples": int(dataset_points.shape[0]),
            **boundary_metrics,
        },
    )


def select_best_candidate(reports: Iterable[SpotterCandidateReport]) -> SpotterCandidateReport:
    reports = list(reports)
    if not reports:
        raise ValueError("No spotter candidate reports available for selection")
    return min(reports, key=lambda report: (float(report.score), report.name))


def _dataset_from_data_test(data_test: Optional[Dict[str, Any]], *, max_points: int):
    if not data_test or "x" not in data_test or "u" not in data_test:
        return None, None
    points = np.asarray(data_test["x"], dtype=np.float64)
    target = np.asarray(data_test["u"], dtype=np.float64)
    return _sample_dataset(points, target, max_points=max_points)


def _dataset_from_reference(reference_solution: Optional[Dict[str, Any]], *, max_points: int):
    if not reference_solution:
        return None, None
    x_ref = reference_solution.get("x_ref")
    u_ref = reference_solution.get("u_ref")
    if x_ref is None or u_ref is None:
        return None, None

    x_ref = np.asarray(x_ref, dtype=np.float64).reshape(-1)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    y_ref = reference_solution.get("y_ref")
    if y_ref is None:
        points = x_ref.reshape(-1, 1)
        target = u_ref.reshape(points.shape[0], -1)
        return _sample_dataset(points, target, max_points=max_points)

    y_ref = np.asarray(y_ref, dtype=np.float64).reshape(-1)
    grid_x, grid_y = np.meshgrid(x_ref, y_ref)
    points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    target = u_ref.reshape(points.shape[0], -1)
    return _sample_dataset(points, target, max_points=max_points)


def _sample_dataset(points: np.ndarray, target: np.ndarray, *, max_points: int):
    if points.shape[0] <= max_points:
        return points, target
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
    return points[indices], target[indices]


def _boundary_residual(provider, data_train: Optional[Dict[str, Any]]):
    if not data_train:
        return None, {}
    boundary_dict = data_train.get("global_boundary_dict")
    if not boundary_dict:
        return None, {}

    total_weight = 0.0
    weighted_error = 0.0
    metrics: Dict[str, Any] = {}

    for var_idx, entry in boundary_dict.items():
        dirichlet = entry.get("dirichlet", {})
        x_dir = np.asarray(dirichlet.get("x", []), dtype=np.float64)
        u_dir = np.asarray(dirichlet.get("values", []), dtype=np.float64)
        w_dir = np.asarray(dirichlet.get("weights", []), dtype=np.float64)
        if x_dir.size > 0 and u_dir.size > 0:
            pred = np.asarray(provider.predict_output(x_dir), dtype=np.float64)
            pred_var = pred[:, int(var_idx) : int(var_idx) + 1]
            err = np.abs(pred_var - u_dir)
            weights = w_dir if w_dir.size else np.ones_like(err)
            weighted_error += float(np.sum(weights * err))
            total_weight += float(np.sum(weights))
            metrics[f"dirichlet_var_{var_idx}_mae"] = float(np.mean(err))

        periodic_pairs = entry.get("periodic", {}).get("pairs", []) or []
        periodic_errors = []
        for pair in periodic_pairs:
            x_1 = np.asarray(pair.get("x_1", []), dtype=np.float64)
            x_2 = np.asarray(pair.get("x_2", []), dtype=np.float64)
            if x_1.size == 0 or x_2.size == 0:
                continue
            pred_1 = np.asarray(provider.predict_output(x_1), dtype=np.float64)[:, int(var_idx) : int(var_idx) + 1]
            pred_2 = np.asarray(provider.predict_output(x_2), dtype=np.float64)[:, int(var_idx) : int(var_idx) + 1]
            err = np.abs(pred_1 - pred_2)
            weight = float(pair.get("weight", 1.0))
            periodic_errors.append(float(np.mean(err)))
            weighted_error += weight * float(np.sum(err))
            total_weight += weight * float(err.size)
        if periodic_errors:
            metrics[f"periodic_var_{var_idx}_mae"] = float(np.mean(periodic_errors))

    if total_weight <= 0.0:
        return None, {}
    score = weighted_error / total_weight
    metrics["boundary_residual"] = float(score)
    return float(score), metrics


def _composite_score(
    *,
    relative_l2: Optional[float],
    absolute_l2: Optional[float],
    boundary_score: Optional[float],
    boundary_metrics: Dict[str, Any],
    priority_score: float,
    weights: Dict[str, Any],
):
    w_l2 = float(weights.get("relative_l2", 1.0))
    w_boundary = float(weights.get("boundary_residual", 1.0))
    w_priority = float(weights.get("priority", 0.0))
    score = 0.0
    metrics: Dict[str, Any] = dict(boundary_metrics)

    if relative_l2 is not None:
        score += w_l2 * float(relative_l2)
        metrics["relative_l2"] = float(relative_l2)
    else:
        metrics["relative_l2"] = None

    if absolute_l2 is not None:
        metrics["absolute_l2"] = float(absolute_l2)
    else:
        metrics["absolute_l2"] = None

    if boundary_score is not None:
        score += w_boundary * float(boundary_score)
    elif relative_l2 is None:
        score += w_priority * priority_score

    if relative_l2 is None and boundary_score is None:
        score += max(w_priority, 1.0) * priority_score
        metrics["priority"] = priority_score
    elif w_priority:
        score += w_priority * priority_score
        metrics["priority"] = priority_score

    metrics["composite_score"] = float(score)
    metrics["weights"] = {
        "relative_l2": w_l2,
        "boundary_residual": w_boundary,
        "priority": w_priority,
    }
    return float(score), metrics
