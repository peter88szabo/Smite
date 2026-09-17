import importlib.util
import os

import numpy as np

from qchem_interfaces.numerical_hessian import central_difference_hessian


_CALCULATORS = {}


def _candidate_pes_dirs(qcinput):
    pes_name = qcinput.get("pes_name") or qcinput.get("name")
    pes_path = qcinput.get("pes_path")
    path = qcinput.get("path")
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_dir = os.path.abspath(os.path.join(src_dir, ".."))

    candidates = []
    if pes_path:
        candidates.append(pes_path)
        if not os.path.isabs(pes_path):
            candidates.append(os.path.join(src_dir, pes_path))
            candidates.append(os.path.join(project_dir, pes_path))
    if path:
        if pes_name:
            candidates.append(os.path.join(path, pes_name))
            if not os.path.isabs(path):
                candidates.append(os.path.join(src_dir, path, pes_name))
                candidates.append(os.path.join(project_dir, path, pes_name))
        candidates.append(path)
        if not os.path.isabs(path):
            candidates.append(os.path.join(src_dir, path))
            candidates.append(os.path.join(project_dir, path))
    if pes_name:
        candidates.append(os.path.join(os.getcwd(), pes_name))
        candidates.append(os.path.join(os.getcwd(), "..", "peslib", pes_name))
        candidates.append(os.path.join(project_dir, "peslib", pes_name))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        full = os.path.abspath(os.path.expanduser(candidate))
        if full not in seen:
            seen.add(full)
            yield full


def _resolve_pes_dir(qcinput):
    for candidate in _candidate_pes_dirs(qcinput):
        if os.path.isdir(candidate):
            return candidate

    pes_name = qcinput.get("pes_name") or qcinput.get("name")
    path = qcinput.get("path") or qcinput.get("pes_path")
    raise FileNotFoundError(
        "Could not locate PES directory. Provide qcinput['pes_name'] and "
        f"qcinput['path'] or qcinput['pes_path']; got pes_name={pes_name!r}, path={path!r}."
    )


def _load_module(module_path):
    spec = importlib.util.spec_from_file_location("smite_pes_interface", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PES interface module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_calculator(pes_dir, qcinput):
    module_path = os.path.join(pes_dir, "pes_interface.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"PES directory {pes_dir} does not contain pes_interface.py. "
            "Add a PESCalculator class there with energy(q, atoms) and gradient(q, atoms) "
            "or force(q, atoms) methods."
        )

    module = _load_module(module_path)
    if not hasattr(module, "PESCalculator"):
        raise AttributeError(f"{module_path} must define a PESCalculator class")

    return module.PESCalculator(pes_dir=pes_dir, config=dict(qcinput))


def _calculator(qcinput):
    pes_dir = _resolve_pes_dir(qcinput)
    config_key = tuple(sorted((str(k), repr(v)) for k, v in qcinput.items()))
    key = (pes_dir, config_key)
    if key not in _CALCULATORS:
        _CALCULATORS[key] = _build_calculator(pes_dir, qcinput)
    return _CALCULATORS[key]


def PES_Energy(file_wf, q, atoms, qcinput):
    del file_wf
    if qcinput.get("wfu"):
        raise ValueError("PES backend does not support wfu=True because no electronic wavefunction is produced.")
    calc = _calculator(qcinput)
    return float(calc.energy(np.asarray(q, dtype=float), atoms))


def PES_Force(q, atoms, qcinput):
    calc = _calculator(qcinput)
    q = np.asarray(q, dtype=float)

    if hasattr(calc, "force"):
        force = calc.force(q, atoms)
    elif hasattr(calc, "gradient"):
        force = -np.asarray(calc.gradient(q, atoms), dtype=float)
    else:
        raise AttributeError("PESCalculator must define force(q, atoms) or gradient(q, atoms)")

    force = np.asarray(force, dtype=float).reshape(-1)
    if force.shape != q.shape:
        raise ValueError(f"PES force shape {force.shape} does not match coordinate shape {q.shape}")
    return force


def PES_Hessian(q, atoms, qcinput):
    calc = _calculator(qcinput)
    q = np.asarray(q, dtype=float).copy()
    if hasattr(calc, "hessian"):
        hess = np.asarray(calc.hessian(q, atoms), dtype=float)
        ndim = len(q)
        if hess.shape != (ndim, ndim):
            raise ValueError(f"PES hessian shape {hess.shape} does not match expected {(ndim, ndim)}")
        return hess

    return central_difference_hessian(
        q,
        lambda coordinates: -PES_Force(coordinates, atoms, qcinput),
        dx=qcinput.get("hessian_dx", 0.002),
    )
