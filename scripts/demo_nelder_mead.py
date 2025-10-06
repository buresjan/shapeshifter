#!/usr/bin/env python3
"""Minimal Nelder-Mead demo using optilb's convenience façade."""

from __future__ import annotations

import numpy as np

from optilb import DesignSpace, OptimizationProblem


def wavy_valley(x: np.ndarray) -> float:
    """Objective with a well-defined minimum and a mild non-linearity."""

    # Shift the basin so the best point is not at the origin.
    target = np.array([0.25, -0.35])
    offset = np.asarray(x, dtype=float) - target

    # Quadratic core plus a gentle periodic modulation to make the search non-trivial.
    bowl = 1.8 * offset[0] ** 2 + 0.6 * offset[1] ** 2
    ripple = 0.05 * np.sin(4.0 * x[0]) * np.cos(3.5 * x[1])
    return float(bowl + ripple)


def main() -> None:
    space = DesignSpace(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    problem = OptimizationProblem(
        objective=wavy_valley,
        space=space,
        x0=[0.4, 0.4],
        optimizer="nelder-mead",
        normalize=True,
    )

    result = problem.run()
    print(f"best_x={result.best_x}")
    print(f"best_f={result.best_f:.6f}")
    if problem.log:
        print(
            f"optimizer={problem.log.optimizer} nfev={problem.log.nfev} runtime={problem.log.runtime:.3f}s",
        )


if __name__ == "__main__":
    main()
