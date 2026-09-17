"""Plot the surface-hopping branching against the exact quantum result."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np

from tully_models import adiabatic_energies, baeck_an_coupling, exact_coupling

fssh = np.loadtxt(HERE / "fssh_tully_branching.dat")
reference = np.loadtxt(HERE / "reference_exact_quantum.dat")

k_fssh, p_fssh, error = fssh[:, 0], fssh[:, 1], fssh[:, 2]
k_exact, p_exact = reference[:, 0], reference[:, 3]

# Exact value at each momentum the ensemble was run at.
matched = np.array([p_exact[np.argmin(np.abs(k_exact - k))] for k in k_fssh])

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

ax = axes[0]
ax.plot(k_exact, p_exact, "s-", ms=9, lw=2.5, color="crimson",
        label="exact quantum (wavepacket)")
ax.errorbar(k_fssh, p_fssh, yerr=error, fmt="o-", ms=8, lw=2, capsize=4,
            color="tab:blue", label="FSSH, Baeck-An couplings\n(100 trajectories)")
ax.set_xlabel("initial momentum $k$ [a.u.]")
ax.set_ylabel("population on the upper surface")
ax.set_title("Tully model I: branching after the crossing")
ax.set_ylim(-0.03, 1.0)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(k_fssh, p_fssh - matched, yerr=error, fmt="o-", ms=8, lw=2,
            capsize=4, color="tab:blue")
ax.axhline(0.0, color="crimson", lw=2)
ax.fill_between([k_fssh.min(), k_fssh.max()], -0.05, 0.05, color="grey", alpha=0.15,
                label="within 0.05")
ax.set_xlabel("initial momentum $k$ [a.u.]")
ax.set_ylabel("FSSH $-$ exact")
ax.set_title("deviation: under-hopping grows with $k$")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Why: the coupling Baeck-An supplies is a narrow spike, so a faster trajectory
# accumulates less transition probability crossing it.
ax = axes[2]
x = np.linspace(-4.0, 4.0, 2001)
ax.plot(x, exact_coupling("SAC", x), lw=2.5, color="crimson", label="exact $|d_{12}|$")
ax.plot(x, baeck_an_coupling("SAC", x), lw=2.2, ls="--", color="tab:blue",
        label="Baeck-An")
ax.set_xlabel("$x$ [bohr]")
ax.set_ylabel("$|d_{12}|$")
ax.set_title("the cause: coupling reduced to a spike")
ax.set_yscale("log")
ax.set_ylim(1e-3, 3.0)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Smite FSSH against exact quantum dynamics, Tully model I", fontsize=13)
fig.savefig(HERE / "fssh_vs_exact.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("  k        FSSH            exact     difference")
for k, p, e, x_ in zip(k_fssh, p_fssh, error, matched):
    print(f"  {k:6.2f}   {p:.4f} +- {e:.4f}   {x_:.4f}   {p - x_:+.4f}")
print("\n  wrote fssh_vs_exact.png")
