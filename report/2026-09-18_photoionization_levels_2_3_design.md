# Photoionization Levels 2 and 3: physical momentum partitioning and photoelectron recoil

Design document. Status: awaiting review. Nothing implemented yet.

Extends `src/photoionization/` from the existing Levels 0 and 1 to a four-level
ladder. Level 2 replaces the isotropic momentum rescale with one that conserves
linear and angular momentum exactly. Level 3 adds momentum-conserving photon and
photoelectron recoil.

---

## 1. Why

### 1.1 What Level 1 does

`prepare_ionic_state` (`src/photoionization/preparation.py:106`) currently closes
the energy balance by multiplying every Cartesian momentum component by one
common factor,

$$\mathbf{p}^{\text{ion}} = s\,\mathbf{p}^{\text{neutral}}, \qquad
  s = \sqrt{\frac{T_{\text{target}}}{T_n}} .$$

### 1.2 Two independent defects

**Defect A — the rescale is unphysical.** The factor $s$ multiplies the centre-of-mass
translation and the rigid rotation together with the vibrations. Level 1's own
premise is that ionization transfers no momentum; under that premise $\mathbf{P}$
and $\mathbf{L}$ must be invariant. They are not. The README (lines 298-302) records
this honestly rather than claiming conservation, but it remains a modelling error,
and it grows with the amount of translation and rotation in the neutral sample.

**Defect B — no momentum bookkeeping.** The photon arrives with $\mathbf{p}_\gamma$
and the photoelectron departs with $\mathbf{p}_e$. The ion must absorb the
difference. Level 1 never forms that vector, so recoil-induced translation,
rotation and vibration are all absent.

These are independent, so they become two levels rather than one.

### 1.3 The ladder

| Level | Momenta | Conserves $\mathbf{P},\mathbf{L}$ | Recoil |
|---|---|---|---|
| 0 | $\mathbf{p}$ unchanged | yes, trivially | no |
| 1 | $s\,\mathbf{p}$, all $3N$ components | **no** | no |
| 2 | Eckart split; scale the internal part only | **yes, exactly** | no |
| 3 | Level 2 plus $\Delta\mathbf{p}=\mathbf{p}_\gamma-\mathbf{p}_e$ | changes them by the correct amount | yes |

Levels 0 and 1 are unchanged. Existing results remain reproducible.

### 1.4 When Level 3 matters

Recoil energy in the delocalised limit, $E_{\text{rec}} = p_e^2/2M = (m_e/M)\,\varepsilon$,
for H$_2$O:

| $h\nu$ | $\varepsilon$ | $p_e$ [a.u.] | $p_\gamma$ [a.u.] | $E_{\text{rec}}$ on COM | $E_{\text{rec}}$ on one O |
|---|---|---|---|---|---|
| 21 eV | 10 eV | 0.857 | 0.006 | 0.31 meV | 0.34 meV |
| 100 eV | 80 eV | 2.425 | 0.027 | 2.44 meV | 2.74 meV |
| 1 keV | 700 eV | 7.173 | 0.268 | 21.3 meV | 24.0 meV |
| 3 keV | 2.7 keV | 14.087 | 0.805 | 82.2 meV | 92.6 meV |

At VUV/TPEPICO energies recoil is sub-meV, far below a vibrational quantum.
At keV it reaches the size of a soft mode and is an established, measured peak
shift. The photon momentum is negligible below roughly 1 keV and reaches 4-6% of
$p_e$ at a few keV; it is cheap to include, so it is included at Level 3
unconditionally.

### 1.5 Choosing a level

The code never picks a level for you and never inspects the photon energy to
decide one. This section is guidance only.

**The one formula.** The recoil energy handed to the nuclei is

```
E_rec  =  |p_e|^2 / (2 M_eff)  =  (m_e / M_eff) * eps
```

where `eps` is the photoelectron kinetic energy and `M_eff` is **the mass that
actually takes the recoil**: the whole molecule for a delocalised outer-valence
hole, but a *single atom* for a localised core hole. That second case is the one
that matters, and it is why photon energy alone is the wrong criterion.

Recoil energy in meV, COM limit / hole localised on the lightest atom:

| molecule | M [u] | 21 eV | 100 eV | 500 eV | 1 keV | 3 keV |
|---|---|---|---|---|---|---|
| H2    |   2.0 | 5.7 / 11.4 | 27.2 / 54.4 | 136 / 272 | 272 / 544 | 816 / 1633 |
| H2O   |  18.0 | 0.6 / 11.4 |  3.0 / 54.4 |  15 / 272 |  30 / 544 |  91 / 1633 |
| CH4   |  16.0 | 0.7 / 11.4 |  3.4 / 54.4 |  17 / 272 |  34 / 544 | 103 / 1633 |
| CO    |  28.0 | 0.4 /  1.0 |  2.0 /  4.6 |  10 /  23 |  20 /  46 |  59 /  137 |
| C6H6  |  78.1 | 0.2 / 11.4 |  0.7 / 54.4 |   4 / 272 |   7 / 544 |  21 / 1633 |
| CF3I  | 195.9 | 0.1 /  0.6 |  0.3 /  2.9 |   1 /  14 |   3 /  29 |   8 /   87 |

Note the columns: on the lightest atom the number depends only on that atom's
mass, not on the molecule. A hole on a hydrogen gives the same 54 meV at 100 eV
in H2 as in benzene.

**Compare `E_rec` against whatever you actually care about** — your analyser
resolution if you are matching a measured peak, or the vibrational energy scale
if you care about the trajectory. Taking a 50 meV resolution as an example, the
photoelectron energy at which recoil becomes visible is:

| | delocalised (COM) | localised on the lightest atom |
|---|---|---|
| H2   |   184 eV |    92 eV |
| H2O  |  1.6 keV |    92 eV |
| CH4  |  1.5 keV |    92 eV |
| CO   |  2.6 keV |   1.1 keV |
| C6H6 |  7.1 keV |    92 eV |
| CF3I |   18 keV |   1.7 keV |

**Suggested choice.**

* **Level 0** — you have no measured electron or ion energy to impose and want a
  pure Franck–Condon transfer. Takes no energy constraints.
* **Level 1** — only to reproduce results generated before Level 2 existed. It is
  superseded: for the same inputs Level 2 gives the same energy with correct
  momentum. There is no case where Level 1 is the better physics.
* **Level 2** — **the default** whenever you impose a measured energy. Threshold
  and VUV work (TPEPICO, He I/II, `hv` up to ~100 eV) with a delocalised valence
  hole sits here: recoil is a few meV at most and Level 3 would change nothing
  you can measure.
* **Level 3** — when `(m_e/M_eff)*eps` reaches the energy you can resolve. In
  practice: any core-level ionization above ~100 eV where the hole sits on a
  light atom, and anything above ~1 keV regardless. Set `recoil_site` to the
  ionized atom for a core hole; leave it `"com"` for a delocalised valence hole,
  where it reduces to pure translation and excites nothing internally.

**Two things that shift the answer.** A hole on hydrogen or a first-row atom puts
`M_eff` one to two orders of magnitude below the molecular mass, so Level 3 earns
its place far below 1 keV. Conversely a heavy atom carrying the hole (iodine
here) pushes the crossover into the many-keV range. The photon momentum term
`p_gamma = hv/c` is under 1% of `p_e` below ~100 eV and reaches 4-6% at a few
keV; it is always included at Level 3 and never matters below soft X-ray.

---

## 2. The model

All nuclear quantities are in atomic units. $m_e = 1$, $c = 137.035999074$.

### 2.1 Mass-weighted momenta

Define

$$\mathbf{a} = \mathsf{M}^{-1/2}\,\mathbf{p}, \qquad a_{3i+\alpha} = \frac{p_{3i+\alpha}}{\sqrt{m_i}},$$

so that the kinetic energy is a plain Euclidean norm,

$$T = \frac{1}{2}\sum_{i\alpha}\frac{p_{3i+\alpha}^2}{m_i} = \frac{1}{2}\,\lvert\mathbf{a}\rvert^2 .$$

This is the metric in which the partitioning below is orthogonal. It is the reason
the construction is exact rather than approximate.

### 2.2 The Eckart partition

`normalmode/eckart.py` already builds the $3N \times 6$ matrix $\mathsf{B}$ whose
columns are the mass-weighted generators of infinitesimal translation and rotation
about the centre of mass (McIver, *J. Chem. Phys.* **88**, 1988, as cited in that
file):

$$\mathsf{B}_{3i+\alpha,\;\beta} = \sqrt{m_i}\,\delta_{\alpha\beta},
\qquad
\mathsf{B}_{3i+\alpha,\;3+\gamma} = \sqrt{m_i}\;\varepsilon_{\alpha\gamma\beta}\,x_{i\beta},$$

with $\mathbf{x}_i$ measured from the centre of mass. `get_eckart_projector`
returns

$$\mathsf{R} = \mathsf{B}\,\mathsf{B}^{+}, \qquad \mathsf{R}^2 = \mathsf{R} = \mathsf{R}^{\mathsf{T}},$$

using the pseudoinverse, so $\operatorname{rank}\mathsf{R} = 6$ for a nonlinear
molecule and $5$ for a linear one with no special-casing. Split

$$\mathbf{a}^{\text{ext}} = \mathsf{R}\,\mathbf{a}, \qquad
  \mathbf{a}^{\text{int}} = (\mathsf{1}-\mathsf{R})\,\mathbf{a}.$$

Because $\mathsf{R}$ is an orthogonal projector **in the mass-weighted metric**,

$$\mathbf{a}^{\text{ext}}\cdot\mathbf{a}^{\text{int}} = 0
\qquad\Longrightarrow\qquad
T = T^{\text{ext}} + T^{\text{int}}$$

with no cross term, and the internal part carries no momentum at all:

$$\sum_i \sqrt{m_i}\,\mathbf{a}^{\text{int}}_i = \mathbf{0},
\qquad
\sum_i \mathbf{x}_i \times \sqrt{m_i}\,\mathbf{a}^{\text{int}}_i = \mathbf{0}.$$

Verified numerically on H$_2$O: cross term $1.3\times10^{-18}$,
$\lvert\mathbf{P}^{\text{int}}\rvert = 1.1\times10^{-15}$,
$\lvert\mathbf{L}^{\text{int}}\rvert = 1.3\times10^{-15}$, $\mathsf{R}$ idempotent
to $3.3\times10^{-16}$.

**Consequence.** Rescaling $\mathbf{a}^{\text{int}}$ by any factor changes the
vibrational energy and leaves $\mathbf{P}$ and $\mathbf{L}$ bit-for-bit untouched.
That is Level 2.

### 2.3 Energy balance

Unchanged from Levels 0 and 1:

$$h\nu + T_n + V_n \;=\; \varepsilon + T_i + V_i ,$$

$$\boxed{\;T_i = T_n + h\nu - \varepsilon - \Delta V\;}, \qquad
\Delta V = (V_i - V_n) + \delta_{\text{offset}} .$$

$\varepsilon$ is the photoelectron kinetic energy, $\Delta V$ the vertical gap at
the frozen Franck-Condon geometry plus the optional additive ionic PES offset.
$\varepsilon$ and the binding energy $h\nu-\varepsilon$ are sampled by the existing
`EnergyConstraints.sample` machinery, unchanged.

### 2.4 Recoil (Level 3)

$$\mathbf{p}_e = \sqrt{2 m_e \varepsilon}\;\hat{\mathbf{n}}
              = \sqrt{2\varepsilon}\;\hat{\mathbf{n}},
\qquad
\mathbf{p}_\gamma = \frac{h\nu}{c}\,\hat{\mathbf{k}},
\qquad
\Delta\mathbf{p} = \mathbf{p}_\gamma - \mathbf{p}_e .$$

$\hat{\mathbf{n}}$ is sampled isotropically on the sphere. $\hat{\mathbf{k}}$ is the
beam direction, default $\hat{\mathbf{z}}$.

The recoil is shared among the atoms with weights $w_i \ge 0$, $\sum_i w_i = 1$:

$$\mathbf{p}_i \;\longrightarrow\; \mathbf{p}_i + w_i\,\Delta\mathbf{p}.$$

Total linear momentum then changes by exactly $\Delta\mathbf{p}$, as required,
for **any** choice of weights. Angular momentum changes by

$$\Delta\mathbf{L} \;=\; \sum_i \mathbf{x}_i \times w_i\Delta\mathbf{p}
\;=\; \Big(\sum_i w_i \mathbf{x}_i\Big) \times \Delta\mathbf{p},$$

i.e. by the displacement of the **recoil centroid** from the centre of mass,
crossed into $\Delta\mathbf{p}$. This single expression contains the physics of the
three supported modes:

| `recoil_site` | $w_i$ | $\Delta\mathbf{L}$ | Physical limit |
|---|---|---|---|
| `"com"` (default) | $m_i/M$ | $\mathbf{0}$ exactly, since $\sum_i m_i\mathbf{x}_i = \mathbf{0}$ | delocalised outer valence |
| atom index $j$ | $\delta_{ij}$ | $\mathbf{x}_j \times \Delta\mathbf{p}$ | localised core hole |
| weight array | user-supplied | $(\sum_i w_i\mathbf{x}_i)\times\Delta\mathbf{p}$ | e.g. Dyson/Mulliken populations |

Mass-weighted recoil produces pure translation and no internal excitation. A
localised hole torques the molecule and excites vibration. **That asymmetry is the
recoil effect**; it is not a numerical artefact.

### 2.5 Closing the balance: the over-determination and its resolution

Given $\varepsilon$, the energy balance fixes $T_i$ *and* momentum conservation
fixes $\mathbf{p}_i$. These are one constraint too many. Two resolutions exist:

- **(A) $\varepsilon$ is an input; the internal subspace absorbs the difference.**
- **(B) $\varepsilon$ is an output**, from the self-consistent root of
  $\varepsilon = h\nu - \Delta V - \Delta T(\varepsilon)$.

**This design adopts (A).** A measured photoelectron peak is already recoil-shifted,
so imposing the experimental $\varepsilon$ and letting the recoil ride on top is
self-consistent with coincidence data, and it preserves the entire purpose of
Level 1. (B) answers a different question — *predicting* the recoil shift from
first principles — and is recorded in §7 as a non-goal.

Naive (A) would rescale the whole internal part and so scrub away the
recoil-induced vibrational excitation that Level 3 exists to capture. Therefore the
rescale is applied **only to the neutral internal component**, leaving the recoil's
internal component intact:

$$\mathbf{a}^{\text{int}} = s\,\mathbf{a}^{\text{int}}_n + \mathbf{a}^{\text{int}}_r .$$

With $T^{\text{int}}_{\text{target}} = T_i - T^{\text{ext}}$ and

$$A = \lvert\mathbf{a}^{\text{int}}_n\rvert^2, \qquad
  B = \mathbf{a}^{\text{int}}_n\cdot\mathbf{a}^{\text{int}}_r, \qquad
  C = \lvert\mathbf{a}^{\text{int}}_r\rvert^2,$$

the condition $\tfrac12\lvert s\,\mathbf{a}^{\text{int}}_n + \mathbf{a}^{\text{int}}_r\rvert^2 = T^{\text{int}}_{\text{target}}$
is the quadratic

$$A s^2 + 2 B s + \big(C - 2T^{\text{int}}_{\text{target}}\big) = 0,$$

$$\boxed{\; s = \frac{-B + \sqrt{D}}{A}, \qquad
  D = B^2 - A\big(C - 2T^{\text{int}}_{\text{target}}\big) \;}$$

always taking the $+$ root, which is the solution retaining the most of the neutral
internal motion.

**Sign of $s$.** For $C < 2T^{\text{int}}_{\text{target}}$ the $+$ root is positive
and the sense of the neutral vibrational motion is preserved. When the recoil alone
over-shoots the internal budget, $C > 2T^{\text{int}}_{\text{target}}$, the $+$ root
can be negative: the neutral internal motion must oppose the recoil to bring the
energy down. This is **permitted, not rejected**, because $D \ge 0$ means the balance
is satisfiable, and the sign of an internal momentum carries no absolute meaning in a
Franck-Condon ensemble -- $+\mathbf{a}^{\text{int}}_n$ and $-\mathbf{a}^{\text{int}}_n$
are equally represented in the neutral sample, so flipping it introduces no bias. The
value is recorded in `momentum_scale` so the regime is visible.

**Level 2 is this with $\mathbf{a}_r = \mathbf{0}$**, whence $B=C=0$ and

$$s = \sqrt{\frac{2T^{\text{int}}_{\text{target}}}{A}} = \sqrt{\frac{T^{\text{int}}_{\text{target}}}{T^{\text{int}}_n}},$$

Level 1's formula restricted to the vibrational subspace. One kernel serves both
levels; Level 2 is Level 3 with the recoil switched off.

---

## 3. Rejection conditions

The module's established style is to raise rather than silently patch. New
rejections, all `ValueError` with the offending quantity in the message:

1. **Monatomic ion at Level 2 or 3.** $\operatorname{rank}\mathsf{R} = 3 = 3N$;
   the internal subspace is empty and no rescale is possible. Level 3 with
   `recoil_site="com"` is still meaningful (pure translation), so this is rejected
   only when a rescale is actually required, i.e. when
   $\lvert T^{\text{int}}_{\text{target}}\rvert > \text{tolerance}$.
2. **$T^{\text{int}}_{\text{target}} < 0$.** The external motion alone exceeds the
   energy budget. Physically: the recoil carries more energy than the photon can pay
   for at this geometry.
3. **$D < 0$.** The experimental energies demand less internal energy than the
   recoil alone deposits. A genuine physical rejection.
4. **$A = 0$ with $T^{\text{int}}_{\text{target}} \neq C/2$.** The neutral sample has
   no internal motion to scale; cannot manufacture it. Mirrors the existing
   zero-momentum rejection at Level 1.
5. **Malformed `recoil_site`**: negative weights, wrong length, zero sum, atom index
   out of range.
6. **Malformed `photon_direction`**: not three finite components, or zero norm.

The existing global energy-residual assertion
$\lvert(T_i - T_n) + \Delta V + \varepsilon - h\nu\rvert \le$ `energy_tolerance`
is retained unchanged and now serves as an independent check on the whole
construction rather than as its definition.

---

## 4. Code changes

### 4.1 New module: `src/photoionization/recoil.py`

`preparation.py` is 126 lines; adding the recoil machinery inline would roughly
double it and mix two concerns. New module holds:

- `sample_electron_momentum(rng, electron_energy_ev, beta=None, polarization=None)`
  — isotropic now; the `beta`/`polarization` arguments are present in the signature
  and rejected as not-yet-implemented, so adding the dipole distribution later is
  not an interface change.
- `photon_momentum(photon_energy_ev, direction)`
- `recoil_weights(recoil_site, mass)` — resolves `"com"` / atom index / array to a
  validated weight vector.

### 4.2 `src/photoionization/preparation.py`

- `eckart_split(q, p, mass)` → `(a_ext, a_int)` in mass-weighted momenta.
- `solve_internal_scale(a_neutral_int, a_recoil_int, t_target)` → the quadratic of
  §2.5, with the rejections of §3.
- `prepare_ionic_state` gains `level in (0,1,2,3)` and the new keywords
  `recoil_site`, `photon_direction`.
- `IonicInitialState` gains fields **appended with defaults** so nothing existing
  breaks: `scaled_subspace`, `external_kinetic_ev`, `internal_kinetic_ev`,
  `recoil_momentum`, `electron_momentum`, `photon_momentum`, `recoil_weights`,
  `recoil_kinetic_ev`, `recoil_internal_kinetic_ev`.
- **`momentum_scale` is redefined, not duplicated.** It carries the factor applied to
  whichever component the level rescales: $1.0$ at Level 0, the global $3N$ factor at
  Level 1, and the neutral-internal factor $s$ at Levels 2 and 3. The companion field
  `scaled_subspace` takes the value `"none"`, `"all"` or `"internal"` so the record is
  self-describing and no consumer has to infer the meaning from `level`.
- **`momentum_scale` must never be NaN.** `driver.py:182` dumps the launch record with
  `json.dump(..., allow_nan=False)`, so a sentinel NaN for "no single factor applies"
  would raise at write time, after the expensive PES evaluations. Hence the
  `scaled_subspace` tag rather than a NaN.

### 4.3 `driver.py`, `system.py`

Accept `level` 2 and 3; thread `recoil_site` and `photon_direction` through
`tpepico`, `Photoionization.__init__` and `Specify_Photoionization_Sampling`.
Reject `recoil_site`/`photon_direction` at Levels 0-2 the same way Level 0 already
rejects experimental constraints. `initial_states.json` metadata gains
`"recoil_included": true/false` driven by the level rather than hard-coded `false`.

### 4.4 `README.md`

Extend "What the levels do" with Levels 2 and 3, the weight table of §2.4, and the
regime table of §1.4.

---

## 5. Documentation standard for the implementation

Requested explicitly: the model must be legible from the code alone, with the
equations present. Three rules.

**R1 — Module docstring states the model and its limits.** Each new or modified
module opens with the physical model, the metric it works in, and what it does not
claim.

**R2 — Every non-obvious step carries its equation in LaTeX**, in the same notation
as this document, so that code and spec can be read against each other.

**R3 — Assumptions and their failure modes are named**, not implied.

Illustrative target for `preparation.py`:

```python
def solve_internal_scale(a_neutral_int, a_recoil_int, t_target):
    r"""Scale the neutral internal momentum to hit a target internal energy.

    The ionic internal (vibrational) momentum is built as the recoil
    contribution, which is fixed by momentum conservation and must not be
    touched, plus a scaled copy of the neutral internal momentum, which is
    the only free quantity left once the energy balance is imposed:

    .. math::
        \mathbf{a}^{\mathrm{int}}
            = s\,\mathbf{a}^{\mathrm{int}}_{n} + \mathbf{a}^{\mathrm{int}}_{r}

    Requiring :math:`T^{\mathrm{int}} = \tfrac12 |\mathbf{a}^{\mathrm{int}}|^2
    = T^{\mathrm{int}}_{\mathrm{target}}` gives a quadratic in :math:`s`,

    .. math::
        A s^2 + 2 B s + (C - 2 T^{\mathrm{int}}_{\mathrm{target}}) = 0,

    with :math:`A = |\mathbf{a}^{\mathrm{int}}_n|^2`,
    :math:`B = \mathbf{a}^{\mathrm{int}}_n \cdot \mathbf{a}^{\mathrm{int}}_r`,
    :math:`C = |\mathbf{a}^{\mathrm{int}}_r|^2`, solved by

    .. math::
        s = \frac{-B + \sqrt{D}}{A},
        \qquad D = B^2 - A\,(C - 2T^{\mathrm{int}}_{\mathrm{target}}).

    The ``+`` root is the physical one: it is positive whenever
    :math:`C < 2T^{\mathrm{int}}_{\mathrm{target}}`, so the sense of the
    neutral vibrational motion is preserved rather than reversed.

    At Level 2 the recoil is absent, :math:`B = C = 0`, and this reduces to
    :math:`s = \sqrt{T^{\mathrm{int}}_{\mathrm{target}} / T^{\mathrm{int}}_n}`
    -- Level 1's factor, but confined to the vibrational subspace.

    Raises
    ------
    ValueError
        If :math:`D < 0`, meaning the sampled experimental energies leave less
        internal energy than the recoil alone deposits. This is a physical
        rejection, not a numerical one: no choice of ``s`` can satisfy it.
    """
```

and for the partition itself:

```python
    # Work in mass-weighted momenta a = p / sqrt(m), where the kinetic energy is
    # a plain Euclidean norm,
    #
    #     T = 1/2 sum_i p_i^2 / m_i = 1/2 |a|^2 .
    #
    # This is the metric in which the Eckart projector R = B B^+ is *orthogonal*,
    # so the split is exact:
    #
    #     a_ext = R a ,   a_int = (1 - R) a ,   a_ext . a_int = 0
    #     =>  T = T_ext + T_int   with no cross term,
    #
    # and a_int carries neither linear nor angular momentum. Scaling a_int alone
    # therefore changes the vibrational energy while leaving P and L bit-for-bit
    # unchanged -- which is the whole point of Level 2 over Level 1. Doing the
    # same projection in unweighted Cartesian coordinates would NOT be orthogonal
    # and would leak energy across the split.
```

---

## 6. Testing

New tests assert the defining properties, not merely that the code runs.

**Level 2**
1. $\lvert\mathbf{P}^{\text{ion}} - \mathbf{P}^{\text{neutral}}\rvert < 10^{-12}$ and likewise for $\mathbf{L}$, on a random polyatomic with translation *and* rotation present. This is the property Level 1 fails.
2. Level 2 $\equiv$ Level 1 when the neutral sample already has $\mathbf{P}=\mathbf{L}=\mathbf{0}$ (nothing external to protect).
3. Energy residual within `energy_tolerance`.
4. Linear molecule (rank-5 projector) succeeds.
5. Monatomic raises when a rescale is required.

**Level 3**
6. `recoil_site="com"`: $\mathbf{P}$ shifts by exactly $\Delta\mathbf{p}$; $\mathbf{L}$ unchanged to $10^{-12}$.
7. `recoil_site=j`: $\Delta\mathbf{L} = \mathbf{x}_j\times\Delta\mathbf{p}$ to $10^{-12}$.
8. Zero neutral momenta, `"com"`, no rescale: recoil energy equals $p_e^2/2M$ analytically.
9. Recoil-induced internal energy survives the rescale — i.e. with a localised site the internal energy differs from the `"com"` case at the same $\varepsilon$. This is the test that would catch a naive implementation of resolution (A).
10. $D<0$ raises with an informative message.
11. Fixed seed reproduces $\hat{\mathbf{n}}$ and hence the whole launch.

**Regression**
12. Levels 0 and 1 byte-identical to current behaviour on the existing fixtures.

Target: extend `src/tests/test_photoionization.py`, plus a new
`src/tests/test_photoionization_recoil.py` for the recoil kernel in isolation.

---

## 7. Non-goals

Explicitly out of scope, to be stated in the README so they are not mistaken for
oversights:

- **Self-consistent $\varepsilon$** (resolution B of §2.5). Would predict the recoil
  shift from first principles instead of consuming a measured one. A plausible
  Level 4.
- **$\beta$-parameter angular distribution.** The signature hook exists; the
  dipole distribution $\frac{d\sigma}{d\Omega} \propto 1 + \beta P_2(\cos\theta)$
  is not implemented. Isotropic sampling is correct for an angle-integrated
  measurement and averages out over an orientationally random ensemble.
- **Mode-resolved energy deposition.** Rejected on principle, not cost: the
  discrepancy being distributed is an artefact of forcing agreement with a measured
  spectrum, so no physical argument selects a mode. Uniform over the internal
  subspace is the honest choice. Would additionally require an ionic Hessian at the
  Franck-Condon geometry.
- **Electron partial waves, photon spin, post-collision interaction.**
- **Multi-electron / shake-up channels.** Single ionization only, $\Delta q = +1$,
  as enforced today.

---

## 8. Recoil partitioning: open questions and planned work

§2.4 supports `"com"`, a single atom, or an explicit weight array. **Those three
modes do not span the physically relevant cases**, and the label "delocalised
outer valence" used for the mass-weighted mode is the wrong name for it.

### 8.1 Two exact statements

The *translational* part of the recoil energy is independent of the weights,

$$T^{\text{trans}}_{\text{rec}} = \frac{|\Delta\mathbf{p}|^2}{2M}
\qquad\text{for every admissible } \{w_i\},$$

because $\sum_i w_i \Delta\mathbf{p} = \Delta\mathbf{p}$ by construction. The
*total* recoil energy is

$$T_{\text{rec}} = \frac{|\Delta\mathbf{p}|^2}{2}\sum_i \frac{w_i^2}{m_i},$$

which, minimised subject to $\sum_i w_i = 1$, gives $w_i = m_i/M$ with minimum
value $|\Delta\mathbf{p}|^2/2M$. Everything above that minimum is internal and
rotational excitation.

So mass weights are the choice depositing **zero** internal energy — not the
delocalised valence limit. A genuinely delocalised valence hole has populations
fixed by electron density, not by mass, and does not reduce to $w_i = m_i/M$.

H₂O, 1 keV photoelectron, recoil energy in meV:

| weights | total | translation | internal + rot |
|---|---|---|---|
| mass, `m_i/M` (`"com"`) | 30.5 | 30.5 | **0.0** |
| Dyson-like, O .70 / H .15 .15 | 41.3 | 30.5 | 10.8 |
| Dyson-like, O .40 / H .30 .30 | 103.4 | 30.5 | 73.0 |
| all on O (core hole) | 34.3 | 30.5 | 3.8 |
| all on H | 544.2 | 30.5 | **513.8** |

A *partially* delocalised hole excites more internal motion (10.8 meV) than one
fully localised on oxygen (3.8 meV), because the light hydrogens are kicked hard
relative to their mass. The two limiting modes do not bracket the answer.

### 8.2 Dyson-orbital weights — to be implemented

The correct weights are the atomic populations $\rho_i$ of the Dyson orbital,

$$\varphi^{\mathrm{D}}(x) = \sqrt{N}\int \Psi_{\text{neutral}}(x, x_2 \ldots x_N)\,
\Psi^{*}_{\text{ion}}(x_2 \ldots x_N)\, \mathrm{d}x_2 \cdots \mathrm{d}x_N,$$

whose density is the density of the hole created. Two decisions remain open and
**both are intended for a later implementation**.

**(i) Where the populations come from.** Either the caller supplies $\rho_i$ as
the weight array (works today), or the module grows a helper reading a
Mulliken/Löwdin analysis from the backend. The latter couples `photoionization`
to backend-specific output, where today it asks backends only for energies and
forces — and a fitted or ML PES has no populations at all. Kept user-supplied
for now for that reason.

**(ii) Deterministic split or stochastic site.** Given $\rho_i$ there are two
inequivalent readings, and this is not a matter of taste:

- **Deterministic** — $w_i = \rho_i$ on every trajectory.
- **Stochastic** — draw one atom with probability $\rho_i$, place the entire
  recoil on it.

Same mean momentum per atom, same translational energy, but for H₂O with
$\rho = (0.70, 0.15, 0.15)$ at 1 keV the deterministic split deposits 10.8 meV
of internal energy against roughly 157 meV for the stochastic choice
($0.70 \times 3.8 + 0.30 \times 513.8$) — **a factor of fifteen**.

The code currently implements the deterministic reading, and that is the one to
prefer on the argument that photoionization is a single coherent event: the hole
is created in a superposition, and "the electron came from atom A" is an
incoherent classical reading valid only if the hole localises before the nuclei
respond. That is a separate assumption about electronic decoherence and should be
stated explicitly rather than smuggled in through the sampler. **Check against
the recoil literature before fixing the choice**, then state it in the interface
instead of leaving it implicit.

### 8.3 Planned extensions, in order of value

1. **Dyson-orbital recoil weights** (§8.2). The largest physical gap in Level 3.
2. **Self-consistent photoelectron energy** (resolution B of §2.5) — would let
   Level 3 *predict* the recoil peak shift rather than consume a measured one.
   A natural Level 4.
3. **The β angular distribution.** The `beta`/`polarization` hooks already exist
   and are refused, so this adds behaviour without an interface change.

---

## 9. References

To be verified against the literature before any publication use.

- C. Eckart, *Phys. Rev.* **47**, 552 (1935) — the Eckart conditions.
- J. W. McIver, *J. Chem. Phys.* **88** (1988) — the $\mathsf{B}$-matrix construction,
  already cited in `src/normalmode/eckart.py`.
- J. Cooper and R. N. Zare, *J. Chem. Phys.* **48**, 942 (1968) — the $\beta$
  asymmetry parameter, for the deferred angular distribution.
- Recoil-induced vibrational and rotational excitation in X-ray photoemission has an
  established experimental literature (work by T. D. Thomas and by E. Kukk and
  co-workers); exact citations to be filled in when the README text is finalised.
