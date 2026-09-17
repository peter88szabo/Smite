import math
import random

import numpy as np

from utils.cenmass import cenmass
from utils.constants import HARTREE_TO_CM1, HARTREE_TO_KCAL_MOL, HARTREE_TO_KJMOL, R_GAS_HARTREE_PER_K
from sampling.thermal import thermal_rot_quantum_spherical_top
from sampling.polyrotation import add_rotational_momentum


def morse_omega0(beta, De, redmass):
    return beta * math.sqrt(2.0 * De / redmass)


def morse_energy(redmass, nv, jrot, beta, De, re):
    """
    Quasiclassical rovibrational energy of the rotating Morse oscillator.
    Ref.: Porter, Raff, Miller, JCP 63, 2214 (1975), Eq. (48).
    """
    omega0 = morse_omega0(beta, De, redmass)
    erot_rigid = jrot * (jrot + 1.0) / (2.0 * redmass * re * re)
    eharm = omega0 * (nv + 0.5)
    eanharm = eharm * eharm / (4.0 * De)
    enonrig = erot_rigid * erot_rigid / (De * beta * beta * re * re)
    ecoup = 3.0 * (1.0 - 1.0 / (beta * re)) * erot_rigid * eharm / (2.0 * beta * re * De)
    return erot_rigid + eharm - eanharm - enonrig - ecoup


def _morse_action_coefficients(redmass, jrot, beta, De, re):
    """Return coefficients for E(x, J) = Ebase + B*x - x**2/(4*De)."""
    parameters = {
        "redmass": redmass,
        "beta": beta,
        "De": De,
        "re": re,
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Morse {name} must be finite and positive")
    if not np.isfinite(jrot) or jrot < 0.0:
        raise ValueError("Morse rotational quantum number must be finite and non-negative")

    omega0 = morse_omega0(beta, De, redmass)
    erot_rigid = jrot * (jrot + 1.0) / (2.0 * redmass * re * re)
    enonrig = erot_rigid * erot_rigid / (De * beta * beta * re * re)
    coupling = (
        3.0
        * (1.0 - 1.0 / (beta * re))
        * erot_rigid
        / (2.0 * beta * re * De)
    )
    return omega0, erot_rigid - enonrig, 1.0 - coupling


def morse_action_for_energy(redmass, target_energy, jrot, beta, De, re):
    """Invert the rotating-Morse energy on its physical lower branch.

    target_energy is the requested total rovibrational energy relative to the
    bottom of the Morse well. The returned quasiclassical nvib can be
    non-integral but always reproduces that energy.
    """
    target_energy = float(target_energy)
    if not np.isfinite(target_energy):
        raise ValueError("Requested Morse energy must be finite")

    omega0, energy_base, linear_coefficient = _morse_action_coefficients(
        redmass, jrot, beta, De, re
    )
    if linear_coefficient <= 0.0:
        raise ValueError(
            "The requested rotational state has no physical lower Morse-action branch"
        )

    branch_maximum = energy_base + De * linear_coefficient * linear_coefficient
    bound_maximum = min(float(De), branch_maximum)
    tolerance = 1.0e-12 * max(1.0, abs(De), abs(bound_maximum))
    if target_energy < energy_base - tolerance:
        raise ValueError(
            f"Requested Morse energy {target_energy:.12g} is below the J={jrot:g} "
            f"rotational minimum {energy_base:.12g}"
        )
    if target_energy >= bound_maximum - tolerance:
        raise ValueError(
            f"Requested Morse energy {target_energy:.12g} is not on the bound lower "
            f"branch (maximum {bound_maximum:.12g})"
        )

    discriminant = (
        linear_coefficient * linear_coefficient
        - (target_energy - energy_base) / De
    )
    if discriminant < -tolerance:
        raise ValueError("Requested Morse energy has no real action solution")
    action_energy = 2.0 * De * (
        linear_coefficient - math.sqrt(max(0.0, discriminant))
    )
    nvib = action_energy / omega0 - 0.5

    recovered = morse_energy(redmass, nvib, jrot, beta, De, re)
    if not math.isclose(recovered, target_energy, rel_tol=1.0e-11, abs_tol=1.0e-12):
        raise RuntimeError(
            "Failed to invert the rotating-Morse energy: "
            f"requested={target_energy:.12g}, recovered={recovered:.12g}"
        )
    return nvib


def _is_bound_morse_state(redmass, nvib, jrot, beta, De, re):
    """Whether a state lies on the bound lower branch used by the sampler."""
    if not np.isfinite(nvib) or not np.isfinite(jrot) or nvib < 0.0 or jrot < 0.0:
        return False

    omega0, _energy_base, linear_coefficient = _morse_action_coefficients(
        redmass, jrot, beta, De, re
    )
    action_energy = omega0 * (nvib + 0.5)
    if linear_coefficient - action_energy / (2.0 * De) <= 0.0:
        return False

    energy = morse_energy(redmass, nvib, jrot, beta, De, re)
    tolerance = 1.0e-12 * max(1.0, abs(De))
    if not np.isfinite(energy) or energy < -tolerance or energy >= De - tolerance:
        return False

    a, b, c = prmconst(jrot * (jrot + 1.0), beta, De, re, energy, redmass)
    discriminant = b * b - 4.0 * a * c
    if a >= 0.0 or discriminant <= 0.0:
        return False
    # This keeps xi positive for every uniformly sampled phase.
    return b - math.sqrt(discriminant) > tolerance


def enumerate_bound_morse_states(
    redmass,
    beta,
    De,
    re,
    *,
    fixed_nvib=None,
    fixed_jrot=None,
):
    """Enumerate bound states of the implemented rotating-Morse Hamiltonian."""
    omega0, _energy_base, _linear_coefficient = _morse_action_coefficients(
        redmass, 0.0, beta, De, re
    )
    vibrational_limit = max(0, int(math.floor(2.0 * De / omega0 - 0.5)))
    inertia = redmass * re * re
    rotational_limit = max(
        0,
        int(math.floor(0.5 * (math.sqrt(1.0 + 8.0 * inertia * De) - 1.0))),
    )

    if fixed_nvib is None:
        vibrational_states = range(vibrational_limit + 1)
    else:
        fixed_nvib = float(fixed_nvib)
        if not np.isfinite(fixed_nvib) or fixed_nvib < 0.0:
            raise ValueError("Fixed Morse vibrational quantum number must be non-negative")
        vibrational_states = (fixed_nvib,)

    if fixed_jrot is None:
        rotational_states = range(rotational_limit + 1)
    else:
        fixed_jrot = float(fixed_jrot)
        if not np.isfinite(fixed_jrot) or fixed_jrot < 0.0:
            raise ValueError("Fixed Morse rotational quantum number must be non-negative")
        rotational_states = (fixed_jrot,)

    state_count = len(vibrational_states) * len(rotational_states)
    if state_count > 2_000_000:
        raise ValueError(
            "Morse canonical state space is too large; check De, beta, and masses"
        )

    states = []
    for jrot in rotational_states:
        for nvib in vibrational_states:
            if not _is_bound_morse_state(redmass, nvib, jrot, beta, De, re):
                continue
            energy = morse_energy(redmass, nvib, jrot, beta, De, re)
            degeneracy = 2.0 * jrot + 1.0
            states.append((nvib, jrot, energy, degeneracy))

    if not states:
        constraints = []
        if fixed_nvib is not None:
            constraints.append(f"nvib={fixed_nvib:g}")
        if fixed_jrot is not None:
            constraints.append(f"J={fixed_jrot:g}")
        suffix = f" for {', '.join(constraints)}" if constraints else ""
        raise ValueError(f"No bound rotating-Morse states exist{suffix}")
    return states


def sample_thermal_morse_state(
    temperature,
    redmass,
    beta,
    De,
    re,
    *,
    fixed_nvib=None,
    fixed_jrot=None,
):
    """Sample the canonical bound-state distribution of the Morse model."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Morse sampling temperature must be finite and positive")
    states = enumerate_bound_morse_states(
        redmass,
        beta,
        De,
        re,
        fixed_nvib=fixed_nvib,
        fixed_jrot=fixed_jrot,
    )

    rt = R_GAS_HARTREE_PER_K * temperature
    log_weights = np.array(
        [math.log(state[3]) - state[2] / rt for state in states],
        dtype=float,
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total_weight = float(np.sum(weights))
    threshold = random.uniform(0.0, total_weight)
    cumulative = 0.0
    for state, weight in zip(states, weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return state[0], state[1], state[2]
    state = states[-1]
    return state[0], state[1], state[2]


def prmconst(angmom2, beta, De, re, Enj, redmass):
    """
    Parameters a, b, c for the sampled Morse variable xi = exp[-beta(r-re)].
    Ref.: Porter, Raff, Miller, JCP 63, 2214 (1975), Eq. (6).
    """
    gg = 1.0 / (beta * re)

    ap = (1.0 - 3.0 * gg * (1.0 - gg)) / (2.0 * redmass * re * re)
    bp = (1.0 - 1.5 * gg) * gg * 2.0 / (redmass * re * re)
    cp = gg * (1.0 - 3.0 * gg) / (2.0 * redmass * re * re)

    a = Enj - De - ap * angmom2
    b = 2.0 * De - bp * angmom2
    c = -De + cp * angmom2

    return a, b, c


def rprsetup(a, b, c, beta, re, redmass):
    """
    Sample the relative Morse coordinate r and conjugate momentum pr
    using the phase-space construction of Porter, Raff, and Miller.
    """
    qn = 2.0 * math.pi * random.uniform(0.0, 1.0)

    discr = b * b - 4.0 * a * c
    if discr <= 0.0:
        raise ValueError("Invalid Morse sampling parameters: b^2 - 4ac must be positive")

    csi = -2.0 * a / (b + math.sin(qn) * math.sqrt(discr))
    if csi <= 0.0:
        raise ValueError("Invalid Morse sampling parameters: sampled Morse variable must be positive")

    r = re - math.log(csi) / beta

    pr_sq = 2.0 * redmass * (a + b * csi + c * csi * csi)
    if pr_sq < -1.0e-12:
        raise ValueError("Invalid Morse sampling parameters: sampled pr^2 became negative")
    pr = math.sqrt(max(pr_sq, 0.0))

    if random.uniform(0.0, 1.0) < 0.5:
        pr = -pr

    return r, pr


def qpsetup(mass, r, pr):
    """
    Build Cartesian q/p for a diatom placed along the x-axis,
    then shift the COM to the origin.
    """
    q1 = np.array([r, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0])

    p1 = np.array([pr, 0.0, 0.0])
    p2 = np.array([-pr, 0.0, 0.0])

    q = np.append(q1, q2)
    p = np.append(p1, p2)

    return cenmass(q, p, mass)


def sample_diatom_rotational_state(rot_modes, inertia, jrot_override=None):
    sampling_mode = rot_modes[0][0]
    excitation = rot_modes[0][1]

    print("\n-------------------------------------------------------------------------")
    print("Diatom Rotational Sampling:")
    if sampling_mode == 'Q':
        jrot = 0.0 if excitation is None else excitation
        print(f"Fix Quantum Number: Jrot = {jrot}")
    elif sampling_mode == 'T':
        print("Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        if jrot_override is None:
            RT = R_GAS_HARTREE_PER_K * excitation
            jrot = thermal_rot_quantum_spherical_top(RT, inertia)
        else:
            jrot = jrot_override
        print(f"jrot = {jrot}")
    else:
        raise ValueError("sampling_mode must be 'Q' or 'T'")

    if jrot_override is not None:
        jrot = jrot_override
    jrot = float(jrot)
    if not np.isfinite(jrot) or jrot < 0.0:
        raise ValueError("Rotational quantum number J must be finite and non-negative")

    angmomabs = math.sqrt(jrot * (jrot + 1.0))
    angle = random.uniform(0.0, 2.0 * math.pi)
    angmom = np.array([0.0, angmomabs * math.sin(angle), angmomabs * math.cos(angle)])
    erot = jrot * (jrot + 1.0) / (2.0 * inertia)
    print(f"Erot = {erot * HARTREE_TO_CM1:>12.2f} cm-1  {erot * HARTREE_TO_KJMOL:>12.3f} kJ/mol")
    print("-------------------------------------------------------------------------\n")

    return jrot, angmom, erot


def _print_morse_vibrational_summary(sampling_mode, excitation, nvib, ezero, evib):
    print("\n-------------------------------------------------------------------------")
    print("Diatom Morse Vibrational Sampling:")
    if sampling_mode == 'Q':
        print(f"Fix Quantum Number: nvib = {nvib}")
    elif sampling_mode == 'T':
        print("Thermal Sampling:")
        print(f"Temp = {excitation:>12.2f} K")
        print(f"nvib = {nvib}")
    elif sampling_mode == 'E':
        print("Fixed Energy Sampling:")
        print(f"Effective nvib = {nvib:.6f}")

    print(f"\n{'traj index:':15} {-999} {'     Ezero':15} {'      Evib':15} {'      Eexc':15}")
    print(f"{'kcal/mol -->':15} {ezero*HARTREE_TO_KCAL_MOL:15.3f} {evib*HARTREE_TO_KCAL_MOL:15.3f} {(evib-ezero)*HARTREE_TO_KCAL_MOL:15.3f}")
    print(f"{'kJ/mol   -->':15} {ezero*HARTREE_TO_KJMOL:15.3f} {evib*HARTREE_TO_KJMOL:15.3f} {(evib-ezero)*HARTREE_TO_KJMOL:15.3f}")
    print(f"{'cm-1     -->':15} {ezero*HARTREE_TO_CM1:15.3f} {evib*HARTREE_TO_CM1:15.3f} {(evib-ezero)*HARTREE_TO_CM1:15.3f}")
    print("-------------------------------------------------------------------------\n")


def diatom_vibration_morse_sampling(vib_modes, rot_modes, req, beta, De, mass):
    """
    Sample a rotating Morse oscillator in atomic units following
    Porter, Raff, and Miller (JCP 63, 2214, 1975).

    The sampled relative coordinate/momentum depends on the rotational
    quantum number, so the rovibrational state is built in one coupled step.
    """
    if len(mass) != 2:
        raise ValueError("Lengths of mass vector in diatom Morse sampling must be 2")

    redmass = mass[0] * mass[1] / (mass[0] + mass[1])
    inertia_eq = redmass * req * req

    sampling_mode = vib_modes[0][1]
    excitation = vib_modes[0][2]
    rotational_mode = rot_modes[0][0]
    rotational_excitation = rot_modes[0][1]
    nvib = None
    jrot = None

    if sampling_mode == 'E' and rotational_mode == 'T':
        raise ValueError(
            "Fixed total Morse energy requires a fixed rotational J; "
            "thermal J and fixed total energy do not define one canonical ensemble"
        )

    if sampling_mode == 'T':
        if rotational_mode == 'T':
            if not math.isclose(
                float(excitation),
                float(rotational_excitation),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "Coupled thermal Morse vibration and rotation require one temperature"
                )
            nvib, jrot, _sampled_energy = sample_thermal_morse_state(
                excitation, redmass, beta, De, req
            )
        elif rotational_mode == 'Q':
            fixed_jrot = 0.0 if rotational_excitation is None else rotational_excitation
            nvib, jrot, _sampled_energy = sample_thermal_morse_state(
                excitation,
                redmass,
                beta,
                De,
                req,
                fixed_jrot=fixed_jrot,
            )
        else:
            raise ValueError("Morse rotational sampling mode must be 'Q' or 'T'")
    elif rotational_mode == 'T':
        if sampling_mode != 'Q':
            raise ValueError("Thermal Morse rotation can only accompany Q or T vibration")
        nvib, jrot, _sampled_energy = sample_thermal_morse_state(
            rotational_excitation,
            redmass,
            beta,
            De,
            req,
            fixed_nvib=excitation,
        )
    elif rotational_mode == 'Q':
        jrot = 0.0 if rotational_excitation is None else float(rotational_excitation)
    else:
        raise ValueError("Morse rotational sampling mode must be 'Q' or 'T'")

    jrot, angmom, erot = sample_diatom_rotational_state(
        rot_modes, inertia_eq, jrot_override=jrot
    )

    if sampling_mode == 'Q':
        if nvib is None:
            nvib = float(excitation)
    elif sampling_mode == 'T':
        pass
    elif sampling_mode == 'E':
        nvib = morse_action_for_energy(
            redmass, excitation, jrot, beta, De, req
        )
    else:
        raise ValueError("sampling_mode must be 'Q', 'T', or 'E' for Morse diatom sampling")

    if not _is_bound_morse_state(redmass, nvib, jrot, beta, De, req):
        raise ValueError(
            f"The requested rotating-Morse state (nvib={nvib:g}, J={jrot:g}) "
            "is not a bound state supported by the phase-space sampler"
        )

    ezero = morse_energy(redmass, 0.0, jrot, beta, De, req)
    Enj = morse_energy(redmass, nvib, jrot, beta, De, req)
    _print_morse_vibrational_summary(sampling_mode, excitation, nvib, ezero, Enj)
    a, b, c = prmconst(jrot * (jrot + 1.0), beta, De, req, Enj, redmass)
    r, pr = rprsetup(a, b, c, beta, req, redmass)
    q, p = qpsetup(mass, r, pr)

    inertia = redmass * r * r
    angvel = np.array([0.0, angmom[1] / inertia, angmom[2] / inertia])
    p = add_rotational_momentum(angvel, mass, q, p)
    q, p = cenmass(q, p, mass)

    return q, p, angmom, inertia, jrot, nvib, Enj
