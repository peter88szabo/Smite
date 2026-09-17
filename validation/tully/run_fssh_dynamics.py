"""Surface hopping on Tully's model I, against the exact quantum result.

Drives Smite's own integrator and FSSH propagator directly rather than going
through ``run_trajectory``, whose stopping conditions are written for molecular
collisions and do not apply to a single particle in one dimension. Everything
that matters -- the velocity Verlet step, the Baeck-An couplings, the amplitude
propagation, hopping and velocity rescaling -- is the package's own code.

The particle starts on the lower adiabatic surface moving right and the run ends
when it leaves the interaction region. What is compared is the fraction of
trajectories ending on the upper surface, against the exact quantum populations
in ``reference_exact_quantum.dat``.

    python run_fssh_dynamics.py [--trajectories N] [--cores N]
"""

from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np

MASS = 2000.0          # Tully's reduced mass for these models
X_START = -6.0
X_STOP = 10.0
DT = 1.0               # atomic units
MAX_STEPS = 40000


def run_one(arguments):
    """One trajectory. Returns the surface it ends on, and where."""
    momentum, seed, model, de_corr = arguments

    from core.molecule import Molecule
    from dynamics.integrator_driver import apply_integrator
    from dynamics.quantum_driver import (
        apply_quantum_integrator,
        initialize_quantum_propagator,
    )

    surfaces = [
        {"qchem": "PES", "pes_path": str(HERE / f"Tully_{model}"),
         "state": state, "wfu": False}
        for state in (0, 1)
    ]

    np.random.seed(seed)

    molecule = Molecule(["X"], np.array([MASS]),
                        np.array([X_START, 0.0, 0.0]),
                        np.array([momentum, 0.0, 0.0]))
    molecule.fname = "tully"
    molecule.qchem = dict(surfaces[0])
    molecule.num_states, molecule.active_state = 2, 0

    propagator = initialize_quantum_propagator(
        "fssh", 2, 0, atoms=["X"], state_qcinput=surfaces,
        de_cutoff=0.5, de_corr=de_corr, n_substeps=20,
    )

    steps = 0
    while abs(molecule.q[0]) < X_STOP and steps < MAX_STEPS:
        apply_integrator(molecule, "verlet", DT)
        apply_quantum_integrator(molecule, "fssh", DT, dtq=None,
                                 propagator=propagator)
        steps += 1

    return molecule.active_state, float(molecule.q[0]), steps


def run_ensemble(momentum, n_trajectories, pool, model="SAC", de_corr=0.0):
    work = [(momentum, 1000 * int(momentum) + i, model, de_corr)
            for i in range(n_trajectories)]
    if pool:
        # imap_unordered so each finished trajectory reports immediately,
        # rather than the whole block going silent until it completes.
        results = []
        for done, item in enumerate(pool.imap_unordered(run_one, work), 1):
            results.append(item)
            if done % 10 == 0 or done == n_trajectories:
                print(f"      k={momentum:6.2f}  {done:4d}/{n_trajectories}",
                      flush=True)
    else:
        results = [run_one(w) for w in work]

    states = np.array([r[0] for r in results])
    positions = np.array([r[1] for r in results])

    transmitted = positions > 0.0
    return {
        "momentum": momentum,
        "upper": float(np.mean(states == 1)),
        "lower": float(np.mean(states == 0)),
        "transmitted": float(np.mean(transmitted)),
        "n": n_trajectories,
        # Binomial standard error on the branching fraction.
        "error": float(np.sqrt(np.mean(states == 1) * (1 - np.mean(states == 1))
                               / n_trajectories)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=200)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--model", default="SAC")
    parser.add_argument("--de-corr", type=float, default=0.0)
    args = parser.parse_args()

    reference = np.loadtxt(HERE / "reference_exact_quantum.dat")
    momenta = sorted(set(list(reference[:, 0]) + [8.0, 12.0, 20.0, 30.0]))

    print(f"Tully model {args.model}: {args.trajectories} trajectories per momentum, "
          f"{args.cores} cores\n", flush=True)
    open("fssh_tully_branching.partial.dat", "w").close()

    import multiprocessing as mp
    pool = mp.Pool(args.cores) if args.cores > 1 else None
    try:
        results = []
        for momentum in momenta:
            outcome = run_ensemble(momentum, args.trajectories, pool,
                                   args.model, args.de_corr)
            results.append(outcome)
            exact = reference[np.isclose(reference[:, 0], momentum), 3]
            note = f"   exact {exact[0]:.4f}" if len(exact) else ""
            print(f"  k = {momentum:6.2f}   P(upper) = {outcome['upper']:.4f} "
                  f"+- {outcome['error']:.4f}{note}", flush=True)
            # Written as each momentum finishes, so a long run is never all or
            # nothing and progress is visible while it is still going.
            with open("fssh_tully_branching.partial.dat", "a") as handle:
                handle.write(f"{momentum:.6f} {outcome['upper']:.6f} "
                             f"{outcome['error']:.6f} {outcome['transmitted']:.6f}\n")
    finally:
        if pool:
            pool.close()
            pool.join()

    rows = [(r["momentum"], r["upper"], r["error"], r["transmitted"]) for r in results]
    np.savetxt("fssh_tully_branching.dat", np.asarray(rows), fmt="%.8e",
               header="k_ini fssh_upper standard_error transmitted")
    print("\n  wrote fssh_tully_branching.dat")

    plot(results, reference, args)


def plot(results, reference, args):
    import matplotlib.pyplot as plt

    momenta = np.array([r["momentum"] for r in results])
    upper = np.array([r["upper"] for r in results])
    error = np.array([r["error"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(momenta, upper, yerr=error, fmt="o-", lw=2, capsize=4,
                label=f"FSSH, {args.trajectories} trajectories")
    ax.plot(reference[:, 0], reference[:, 3], "s", ms=11, mfc="none", mew=2.5,
            color="crimson", label="exact quantum")
    ax.set_xlabel("initial momentum $k$ [a.u.]")
    ax.set_ylabel("population on the upper surface")
    ax.set_title(f"Tully {args.model}: branching after the crossing")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    shared = [i for i, k in enumerate(momenta)
              if np.any(np.isclose(reference[:, 0], k))]
    if shared:
        k_shared = momenta[shared]
        fssh_shared = upper[shared]
        error_shared = error[shared]
        exact_shared = np.array([
            reference[np.isclose(reference[:, 0], k), 3][0] for k in k_shared
        ])
        ax.errorbar(k_shared, fssh_shared - exact_shared, yerr=error_shared,
                    fmt="o", ms=9, capsize=4, color="crimson")
        ax.axhline(0.0, color="k", lw=1)
        span = max(0.05, 1.3 * np.max(np.abs(fssh_shared - exact_shared)))
        ax.set_ylim(-span, span)
    ax.set_xlabel("initial momentum $k$ [a.u.]")
    ax.set_ylabel("FSSH $-$ exact")
    ax.set_title("deviation where an exact result exists")
    ax.grid(alpha=0.3)

    fig.savefig("fssh_tully_branching.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fssh_tully_branching.png")


if __name__ == "__main__":
    main()
