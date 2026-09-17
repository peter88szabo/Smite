from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from utils.constants import (
    ATOMIC_MASS_GMOL_TO_AU,
    AU_ANGULAR_FREQUENCY_TO_CM1,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_KCAL_MOL,
    HARTREE_TO_KJMOL,
)


PI = math.pi
RGAS_AU = 8.31446261815324 / 1000.0 / HARTREE_TO_KJMOL
CM1_TO_K = 1.43877
CM1_TO_HARTREE = 4.55635e-6
CM1_TO_KCAL = 2.85914e-3
PASCAL_TO_AU = 1.0e-13 / 2.9421912
AMU_TO_ELECMASS = 1836.15267343
HPLANCK_AU_SQ = (2.0 * PI) ** 2
J_PER_HARTREE_PER_MOL = HARTREE_TO_KJMOL * 1000.0

PLANCK_SI = 6.62607015e-34
BOLTZMANN_SI = 1.380649e-23
RGAS_SI = 8.31446261815324
CLIGHT_SI = 2.99792458e8

GRIMME_BAV_SI = 1.0e-44
GRIMME_ALPHA = 4.0

# h / (8 pi^2 c) converted to cm^-1 for I in amu Angstrom^2.
ROTCONST_CM1_PER_AMU_ANG2 = 16.857629206


@dataclass
class ThermoResults:
    pfelec: float = 1.0
    pftrans: float = 1.0
    pfrot: float = 1.0
    pfvib: float = 1.0
    pftot: float = 1.0

    uelec: float = 0.0
    utrans: float = 0.0
    urot: float = 0.0
    uvib: float = 0.0
    utherm: float = 0.0
    utot: float = 0.0

    helec: float = 0.0
    htrans: float = 0.0
    hrot: float = 0.0
    hvib: float = 0.0
    htherm: float = 0.0
    htot: float = 0.0

    felec: float = 0.0
    ftrans: float = 0.0
    frot: float = 0.0
    fvib: float = 0.0
    ftherm: float = 0.0
    ftot: float = 0.0

    gelec: float = 0.0
    gtrans: float = 0.0
    grot: float = 0.0
    gvib: float = 0.0
    gtherm: float = 0.0
    gtot: float = 0.0

    selec: float = 0.0
    strans: float = 0.0
    srot: float = 0.0
    svib: float = 0.0
    stherm: float = 0.0
    stot: float = 0.0

    cvelec: float = 0.0
    cvtrans: float = 0.0
    cvrot: float = 0.0
    cvvib: float = 0.0
    cvtherm: float = 0.0
    cvtot: float = 0.0

    cpelec: float = 0.0
    cptrans: float = 0.0
    cprot: float = 0.0
    cpvib: float = 0.0
    cptherm: float = 0.0
    cptot: float = 0.0

    zpe: float = 0.0


def _mass_au_to_amu(mass):
    return np.asarray(mass, dtype=float) / ATOMIC_MASS_GMOL_TO_AU


def center_coordinates(mass_amu, q_bohr):
    mass_amu = np.asarray(mass_amu, dtype=float)
    xyz = np.asarray(q_bohr, dtype=float).reshape((-1, 3))
    center = np.sum(xyz * mass_amu[:, None], axis=0) / np.sum(mass_amu)
    return xyz - center


def inertia_tensor_amu_ang2(q_bohr, mass_amu):
    xyz_ang = center_coordinates(mass_amu, q_bohr) * BOHR_TO_ANGSTROM
    tensor = np.zeros((3, 3), dtype=float)
    for r, mass in zip(xyz_ang, mass_amu):
        tensor += mass * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))
    return 0.5 * (tensor + tensor.T)


def rotational_constants_cm1(q_bohr, mass, *, mass_is_au=True, tol=1.0e-10):
    mass_amu = _mass_au_to_amu(mass) if mass_is_au else np.asarray(mass, dtype=float)
    tensor = inertia_tensor_amu_ang2(q_bohr, mass_amu)
    moments = np.linalg.eigvalsh(tensor)
    moments = np.where(np.abs(moments) < tol, 0.0, moments)
    constants = np.zeros(3, dtype=float)
    mask = moments > tol
    constants[mask] = ROTCONST_CM1_PER_AMU_ANG2 / moments[mask]
    return constants


def freqs_au_to_cm1(freqs_au):
    return np.asarray(freqs_au, dtype=float) * AU_ANGULAR_FREQUENCY_TO_CM1


def _all_electronic(thermo, multiplicity, temp):
    rt = RGAS_AU * temp
    pf = float(multiplicity) if multiplicity and multiplicity > 0.0 else 1.0
    free_energy = -rt * math.log(pf)
    entropy = -free_energy / temp
    thermo.pfelec = pf
    thermo.felec = free_energy
    thermo.gelec = free_energy
    thermo.selec = entropy


def _all_translation(thermo, mass_amu_total, pressure, temp):
    rt = RGAS_AU * temp
    mass = mass_amu_total * AMU_TO_ELECMASS
    lam = math.sqrt(2.0 * PI * mass * rt / HPLANCK_AU_SQ) ** 3
    volume = rt / (pressure * PASCAL_TO_AU)
    pf = math.e * lam * volume
    free_energy = -rt * math.log(pf)
    internal_energy = 1.5 * rt
    enthalpy = 2.5 * rt
    entropy = (internal_energy - free_energy) / temp
    gibbs = enthalpy - temp * entropy

    thermo.pftrans = pf
    thermo.ftrans = free_energy
    thermo.utrans = internal_energy
    thermo.htrans = enthalpy
    thermo.strans = entropy
    thermo.gtrans = gibbs
    thermo.cvtrans = 1.5 * RGAS_AU
    thermo.cptrans = 2.5 * RGAS_AU


def _all_rotations(thermo, brot_cm1, temp):
    brot = np.asarray(brot_cm1, dtype=float)
    if brot.size == 0:
        return

    rt = RGAS_AU * temp
    sigma = 1.0
    chiral = 1.0
    eps = 1.0e-12
    nonzero = brot[brot > eps]
    has_zero = np.any(brot <= eps)

    if has_zero or nonzero.size <= 1:
        brot_mean = float(np.mean(nonzero)) if nonzero.size else max(float(brot[0]), 1.0e-6)
        theta_r = brot_mean * CM1_TO_K
        pf = (temp / theta_r) * (chiral / sigma)
        dof = 2.0
    else:
        theta = nonzero[:3] * CM1_TO_K
        pf = math.sqrt(PI) * temp**1.5 / math.sqrt(float(np.prod(theta))) * (chiral / sigma)
        dof = 3.0

    free_energy = -rt * math.log(pf)
    internal_energy = 0.5 * dof * rt
    entropy = (internal_energy - free_energy) / temp

    thermo.pfrot = pf
    thermo.frot = free_energy
    thermo.urot = internal_energy
    thermo.hrot = internal_energy
    thermo.srot = entropy
    thermo.grot = free_energy
    thermo.cvrot = 0.5 * dof * RGAS_AU
    thermo.cprot = thermo.cvrot


def _entropy_vib_rrho(omega_cm1, temp):
    if omega_cm1 <= 0.0:
        return 0.0
    x = CM1_TO_K * omega_cm1 / temp
    if x < 1.0e-12:
        return 0.0
    ex = math.exp(x)
    term = x / (ex - 1.0) - math.log(1.0 - math.exp(-x))
    return RGAS_AU * term


def _entropy_free_rotor(omega_cm1, temp, bav_si):
    if omega_cm1 <= 0.0:
        return 0.0
    omega_m1 = omega_cm1 * 100.0
    nu_s1 = CLIGHT_SI * omega_m1
    mu = PLANCK_SI / (8.0 * PI * PI * nu_s1)
    mu_prime = mu * bav_si / (mu + bav_si)
    factor = 8.0 * PI**3 * mu_prime * BOLTZMANN_SI * temp / (PLANCK_SI * PLANCK_SI)
    return RGAS_SI * (0.5 + 0.5 * math.log(factor)) / J_PER_HARTREE_PER_MOL


def _grimme_damp(omega_cm1, freq_cutoff_cm1):
    ratio = freq_cutoff_cm1 / omega_cm1
    return 1.0 / (1.0 + ratio**GRIMME_ALPHA)


def _grimme_entropy_qrrho(omega_cm1, freq_cutoff_cm1, temp):
    if omega_cm1 <= 0.0:
        return 0.0
    weight = _grimme_damp(omega_cm1, freq_cutoff_cm1)
    s_rrho = _entropy_vib_rrho(omega_cm1, temp)
    s_fr = _entropy_free_rotor(omega_cm1, temp, GRIMME_BAV_SI)
    return weight * s_rrho + (1.0 - weight) * s_fr


def _all_vibrations(thermo, freqs_cm1, temp, freq_cutoff):
    rt = RGAS_AU * temp
    for omega_cm1 in np.asarray(freqs_cm1, dtype=float):
        omega_cm1 = float(omega_cm1)
        if omega_cm1 <= 0.0:
            continue

        thermo.zpe += 0.5 * omega_cm1 * CM1_TO_HARTREE
        x = CM1_TO_K * omega_cm1 / temp
        ex = math.exp(x)
        u_mode = omega_cm1 * CM1_TO_HARTREE / (ex - 1.0)
        cv_mode = RGAS_AU * x * x * ex / ((ex - 1.0) * (ex - 1.0))
        if omega_cm1 > freq_cutoff:
            s_mode = _entropy_vib_rrho(omega_cm1, temp)
        else:
            s_mode = _grimme_entropy_qrrho(omega_cm1, freq_cutoff, temp)
        f_mode = u_mode - temp * s_mode
        pf_mode = math.exp(-f_mode / rt)

        thermo.uvib += u_mode
        thermo.hvib += u_mode
        thermo.svib += s_mode
        thermo.fvib += f_mode
        thermo.gvib += f_mode
        thermo.cvvib += cv_mode
        thermo.cpvib += cv_mode
        thermo.pfvib *= pf_mode


def eval_thermo(freqs_cm1, brot_cm1, mass_amu_total, multiplicity, temp, pressure, freq_cutoff):
    thermo = ThermoResults()
    _all_electronic(thermo, multiplicity, temp)
    _all_translation(thermo, mass_amu_total, pressure, temp)
    _all_rotations(thermo, brot_cm1, temp)
    _all_vibrations(thermo, freqs_cm1, temp, freq_cutoff)

    thermo.utherm = thermo.uelec + thermo.utrans + thermo.urot + thermo.uvib
    thermo.htherm = thermo.helec + thermo.htrans + thermo.hrot + thermo.hvib
    thermo.stherm = thermo.selec + thermo.strans + thermo.srot + thermo.svib
    thermo.ftherm = thermo.felec + thermo.ftrans + thermo.frot + thermo.fvib
    thermo.gtherm = thermo.gelec + thermo.gtrans + thermo.grot + thermo.gvib
    thermo.cvtherm = thermo.cvelec + thermo.cvtrans + thermo.cvrot + thermo.cvvib
    thermo.cptherm = thermo.cpelec + thermo.cptrans + thermo.cprot + thermo.cpvib

    thermo.utot = thermo.utherm + thermo.zpe
    thermo.htot = thermo.htherm + thermo.zpe
    thermo.ftot = thermo.ftherm + thermo.zpe
    thermo.gtot = thermo.gtherm + thermo.zpe
    thermo.stot = thermo.stherm
    thermo.cvtot = thermo.cvtherm
    thermo.cptot = thermo.cptherm
    thermo.pftot = thermo.pfelec * thermo.pftrans * thermo.pfrot * thermo.pfvib
    return thermo


def print_thermo(thermo, temp, freq_cutoff, electronic_energy=None):
    def print_uhfg(label, u, h, f, g):
        print(f"{label:<14} {u:>12.6f}  {h:>12.6f}  {f:>12.6f}  {g:>12.6f}")

    def print_scc(label, s, cv, cp):
        print(f"{label:<14} {s:>12.6f}  {cv:>12.6f}  {cp:>12.6f}")

    print()
    print("========================= Thermochemistry =========================")
    print(f"T = {temp:.2f} K")
    print(f"ZPE: {thermo.zpe:>12.6f} Eh  ({thermo.zpe * CM1_TO_KCAL / CM1_TO_HARTREE:>10.3f} kcal/mol)")
    print(f"qRRHO cutoff: {freq_cutoff:.1f} cm-1")
    print()
    print("---- Energy Contributions (Eh) ----")
    print(f"{'':<14} {'U':>12}  {'H':>12}  {'F':>12}  {'G':>12}")
    print_uhfg("Electronic", thermo.uelec, thermo.helec, thermo.felec, thermo.gelec)
    print_uhfg("Trans", thermo.utrans, thermo.htrans, thermo.ftrans, thermo.gtrans)
    print_uhfg("Rot", thermo.urot, thermo.hrot, thermo.frot, thermo.grot)
    print_uhfg("Vib", thermo.uvib, thermo.hvib, thermo.fvib, thermo.gvib)
    print_uhfg("Thermal", thermo.utherm, thermo.htherm, thermo.ftherm, thermo.gtherm)
    print("-" * 66)
    print_uhfg("Total(Eh)", thermo.utot, thermo.htot, thermo.ftot, thermo.gtot)
    print_uhfg(
        "Total(kcal/mol)",
        thermo.utot * HARTREE_TO_KCAL_MOL,
        thermo.htot * HARTREE_TO_KCAL_MOL,
        thermo.ftot * HARTREE_TO_KCAL_MOL,
        thermo.gtot * HARTREE_TO_KCAL_MOL,
    )
    print()
    if electronic_energy is None:
        print("Electronic energy (Eh): n/a")
    else:
        e_elec = float(electronic_energy)
        print(f"Electronic energy (Eh): {e_elec:>12.6f}")
        print(f"Electronic + ZPE (Eh): {e_elec + thermo.zpe:>12.6f}")
        print(f"{'Total+E_elec':<14} {'U':>12}  {'H':>12}  {'F':>12}  {'G':>12}")
        print_uhfg(
            "Eh",
            e_elec + thermo.utot,
            e_elec + thermo.htot,
            e_elec + thermo.ftot,
            e_elec + thermo.gtot,
        )

    print()
    print("---- Entropy & Heat Capacities (Eh/K) ----")
    print(f"{'':<14} {'S':>12}  {'Cv':>12}  {'Cp':>12}")
    print_scc("Electronic", thermo.selec, thermo.cvelec, thermo.cpelec)
    print_scc("Trans", thermo.strans, thermo.cvtrans, thermo.cptrans)
    print_scc("Rot", thermo.srot, thermo.cvrot, thermo.cprot)
    print_scc("Vib", thermo.svib, thermo.cvvib, thermo.cpvib)
    print_scc("Thermal", thermo.stherm, thermo.cvtherm, thermo.cptherm)
    print("-" * 49)
    print_scc("Total", thermo.stot, thermo.cvtot, thermo.cptot)
    print(f"S_total*T = {thermo.stot * temp * HARTREE_TO_KCAL_MOL:>10.3f} kcal/mol")


def thermochemistry_analysis(
    atoms,
    q_bohr,
    mass,
    freqs_au=None,
    freqs_cm1=None,
    *,
    temp=298.15,
    pressure=101325.0,
    multiplicity=1,
    qrrho_cutoff=50.0,
    electronic_energy=None,
    print_report=True,
):
    del atoms
    mass = np.asarray(mass, dtype=float)
    mass_amu = _mass_au_to_amu(mass)
    if freqs_cm1 is None:
        if freqs_au is None:
            raise ValueError("thermochemistry_analysis requires freqs_au or freqs_cm1")
        freqs_cm1 = freqs_au_to_cm1(freqs_au)
    freqs_cm1 = np.asarray(freqs_cm1, dtype=float)
    freqs_cm1 = freqs_cm1[freqs_cm1 > 0.0]
    brot_cm1 = rotational_constants_cm1(q_bohr, mass, mass_is_au=True)
    thermo = eval_thermo(
        freqs_cm1,
        brot_cm1,
        float(np.sum(mass_amu)),
        float(multiplicity) if multiplicity is not None else 1.0,
        float(temp),
        float(pressure),
        float(qrrho_cutoff),
    )
    if print_report:
        print_thermo(thermo, float(temp), float(qrrho_cutoff), electronic_energy=electronic_energy)
    return {
        "thermo": thermo,
        "freqs_cm1": freqs_cm1,
        "rotational_constants_cm1": brot_cm1,
        "mass_amu_total": float(np.sum(mass_amu)),
    }
