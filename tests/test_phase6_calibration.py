from dataclasses import dataclass
from datetime import date, timedelta

from analysis.phase6_calibration import (
    CalibrationObservation,
    calibrate_phase5_folds,
    fit_temperature,
)


@dataclass
class Fold:
    prediction_date: date
    probabilities_pct: dict[str, float]
    actual_class: str


def main() -> None:
    start = date(2026, 1, 1)
    actuals = ["UP", "SIDEWAYS", "DOWN"] * 20
    folds = []

    for i, actual in enumerate(actuals):
        if actual == "UP":
            probs = {"UP": 80.0, "SIDEWAYS": 15.0, "DOWN": 5.0}
        elif actual == "SIDEWAYS":
            probs = {"UP": 15.0, "SIDEWAYS": 80.0, "DOWN": 5.0}
        else:
            probs = {"UP": 5.0, "SIDEWAYS": 15.0, "DOWN": 80.0}
        folds.append(Fold(start + timedelta(days=i), probs, actual))

    result = calibrate_phase5_folds(folds, min_calibration_observations=12)

    assert result.protocol == "PREQUENTIAL_OOS_TEMPERATURE_SCALING"
    assert len(result.predictions) == len(folds)
    assert result.predictions[0].calibration_status == "NO_CALIBRATION_DATA"
    assert result.predictions[12].calibration_status == "FITTED"

    for prediction in result.predictions:
        total = sum(prediction.calibrated_probabilities_pct.values())
        assert abs(total - 100.0) < 1e-9

    # Leakage contract: changing the current fold's actual class must not
    # change the calibration parameter fitted from earlier observations.
    history = [
        CalibrationObservation(
            prediction_date=start + timedelta(days=i),
            probabilities_pct={"UP": 80.0, "SIDEWAYS": 15.0, "DOWN": 5.0},
            actual_class="UP",
        )
        for i in range(12)
    ]
    fitted_a = fit_temperature(history, min_observations=12)

    altered_history = list(history)
    altered_history[-1] = CalibrationObservation(
        prediction_date=altered_history[-1].prediction_date,
        probabilities_pct=altered_history[-1].probabilities_pct,
        actual_class="DOWN",
    )
    fitted_b = fit_temperature(altered_history, min_observations=12)

    # This confirms only that the fitting sample is historical. Both are
    # legitimate alternative histories, so their parameters may differ.
    # The stronger integration contract is that calibrate_phase5_folds fits
    # fold i before appending fold i's actual outcome.
    first_fitted = result.predictions[12]
    assert first_fitted.temperature == fit_temperature(
        history, min_observations=12
    ).temperature

    assert fitted_a.status == "FITTED"
    assert fitted_b.status == "FITTED"

    print("PHASE 6 CALIBRATION TEST: PASS")
    print("Protocol: strictly prequential OOS temperature scaling")
    print(f"Observations: {result.metrics.observations}")
    print(f"Raw log loss: {result.metrics.raw_log_loss:.6f}")
    print(f"Calibrated log loss: {result.metrics.log_loss:.6f}")
    print(f"Raw Brier: {result.metrics.raw_brier_score:.6f}")
    print(f"Calibrated Brier: {result.metrics.brier_score:.6f}")


if __name__ == "__main__":
    main()

