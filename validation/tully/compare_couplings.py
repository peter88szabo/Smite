"""Baeck-An couplings against the exact nonadiabatic coupling.

Smite approximates the coupling from the curvature of the adiabatic gap, because
a machine-learned surface offers no wavefunction to differentiate. Tully's models
are the place to find out what that costs: they are two-state and one
dimensional, so the exact coupling

    d12(x) = -dtheta/dx ,    theta = 1/2 arctan2( 2 V12, V11 - V22 )

is available in closed form and the approximation can be compared with it point
by point. No trajectories are involved, so this isolates the coupling from
everything else in the method.

Run:

    python compare_couplings.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from tully_models import MODELS, adiabatic_energies, baeck_an_coupling, exact_coupling


def compare(model: str, x: np.ndarray) -> dict:
    exact = exact_coupling(model, x)
    approximate = baeck_an_coupling(model, x)

    peak = int(np.argmax(exact))
    appreciable = exact > 0.05 * exact[peak]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(exact > 1.0e-12, approximate / exact, np.nan)

    return {
        "x": x,
        "exact": exact,
        "baeck_an": approximate,
        "ratio": ratio,
        "peak_x": float(x[peak]),
        "peak_exact": float(exact[peak]),
        "peak_baeck_an": float(approximate[peak]),
        "peak_ratio": float(approximate[peak] / exact[peak]),
        "appreciable": appreciable,
        "silent_fraction": float(
            np.mean(approximate[appreciable] == 0.0)
        ),
        "median_ratio": float(np.nanmedian(ratio[appreciable])),
    }


def main():
    x = np.linspace(-10.0, 10.0, 4001)
    results = {model: compare(model, x) for model in MODELS}

    print("Baeck-An coupling against the exact d12, Tully models\n")
    print(f"  {'model':<6} {'at the peak of |d12|':<34} {'over the coupling region':<34}")
    print(f"  {'':<6} {'x':>7} {'exact':>9} {'approx':>9} {'ratio':>6}   "
          f"{'median ratio':>13} {'returns zero':>13}")
    for model, r in results.items():
        print(f"  {model:<6} {r['peak_x']:>7.2f} {r['peak_exact']:>9.4f} "
              f"{r['peak_baeck_an']:>9.4f} {r['peak_ratio']:>6.3f}   "
              f"{r['median_ratio']:>13.3f} {100 * r['silent_fraction']:>12.0f}%")

    print("\n  'returns zero' is the fraction of the region where the exact coupling")
    print("  is appreciable but Baeck-An gives nothing, because the gap curves the")
    print("  wrong way there.\n")

    for model, r in results.items():
        rows = np.column_stack([r["x"], r["exact"], r["baeck_an"], r["ratio"]])
        name = f"tully_{model}_couplings.dat"
        np.savetxt(name, rows, fmt="%.8e", delimiter=" ",
                   header="x_bohr exact_d12 baeck_an_d12 ratio")
        print(f"  wrote {name}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib is absent, so no figure was drawn.")
        return results

    fig, axes = plt.subplots(2, len(MODELS), figsize=(5 * len(MODELS), 8),
                             sharex="col", constrained_layout=True)
    for column, (model, r) in enumerate(results.items()):
        lower, upper = adiabatic_energies(model, r["x"])

        top = axes[0][column]
        top.plot(r["x"], lower, lw=1.5, label="lower")
        top.plot(r["x"], upper, lw=1.5, label="upper")
        top.set_title(f"{model}: {MODELS[model][2]}")
        top.set_ylabel("Energy [Hartree]")
        top.legend(fontsize="small")

        bottom = axes[1][column]
        bottom.plot(r["x"], r["exact"], lw=2, label="exact $|d_{12}|$")
        bottom.plot(r["x"], r["baeck_an"], lw=1.8, ls="--", label="Baeck-An")
        bottom.set_xlabel("$x$ [bohr]")
        bottom.set_ylabel("$|d_{12}|$")
        bottom.set_yscale("log")
        bottom.set_ylim(1.0e-4, max(2.0 * r["peak_exact"], 1.0))
        bottom.legend(fontsize="small")

    fig.suptitle("Baeck-An approximation against the exact nonadiabatic coupling")
    fig.savefig("tully_coupling_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote tully_coupling_comparison.png")

    return results


if __name__ == "__main__":
    main()
