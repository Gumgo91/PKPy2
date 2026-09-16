"""Analytical concentration predictions for single and repeated dosing.

``predict`` accepts explicit ``(time, amount)`` dose events. One-compartment
IV/oral and two-compartment IV use dose-state recurrence for repeated doses;
two-compartment oral uses direct analytical superposition.
"""
import time

import numpy as np

from pkpy2 import predict


def main():
    # 100 mg every 12 h for 30 days -> 60 dose events.
    doses = [(12. * k, 100.) for k in range(60)]
    times = np.linspace(0.1, 60 * 12., 400)

    theta = {
        "1cmt_iv": {"CL": 4., "V": 40.},
        "1cmt_oral": {"CL": 4., "V": 40., "Ka": 1.2, "ALAG": 0.},
        "2cmt_iv": {"CL": 4., "V1": 25., "Q": 8., "V2": 50.},
        "2cmt_oral": {"CL": 4., "V1": 25., "Q": 8., "V2": 50., "Ka": 1.2,
                      "ALAG": 0.},
    }
    for model, parameters in theta.items():
        start = time.perf_counter()
        concentrations = predict(model, parameters, times, doses)
        elapsed = time.perf_counter() - start
        print(f"{model:>10s}  n_doses={len(doses):3d}  n_times={len(times)}  "
              f"max={concentrations.max():.3f} mg/L  ({elapsed * 1e3:.1f} ms "
              f"including first-call JIT compilation)")


if __name__ == "__main__":
    main()
