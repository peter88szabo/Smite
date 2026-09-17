from __future__ import annotations

import copy
import hashlib
import contextlib
import json
import os
import queue as queue_module
import re
import sys
import traceback as traceback_module
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable

from qchem_interfaces.qchem_validation import validate_qchem_input
from sampling.random_seed import set_sampling_seed
from utils.constants import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class TrajectoryContext:
    itraj: int
    trajectory_dir: str
    scratch_dir: str
    traj_file: str
    backfile: str
    initial_conditions_file: str
    nproc_per_job: int
    sampling_seed: int | None = None


@dataclass
class ParallelTrajectoryResult:
    itraj: int
    status: str
    trajectory_dir: str
    traj_file: str
    backfile: str
    start_time: str
    end_time: str | None = None
    error: str | None = None
    traceback: str | None = None
    step: int | None = None
    time_fs: float | None = None
    energy_drift_kjmol: float | None = None
    temperature_K: float | None = None
    rcom_angstrom: float | None = None
    last_update: str | None = None


_PROGRESS_RE = re.compile(
    r"step\s*[:=]\s*(?P<step>\d+)\s+"
    r"t\[fs\]\s*[:=]\s*(?P<time_fs>[-+0-9.eE]+).*?"
    r"dE\[kJ\]\s*[:=]\s*(?P<energy_drift>[-+0-9.eE]+)\s+"
    r"T\[K\]\s*[:=]\s*(?P<temperature>[-+0-9.eE]+)"
    r"(?:\s+Rcom\[A\]\s*[:=]\s*(?P<rcom>[-+0-9.eE]+))?"
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def derive_trajectory_seed(base_seed, itraj):
    """Derive a stable, schedule-independent 32-bit seed for one trajectory."""
    if base_seed is None:
        return None
    payload = f"{int(base_seed)}:{int(itraj)}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=4, person=b"SmiteQCT").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)

def _resolve_base_seed(base_seed):
    if base_seed is None:
        return int.from_bytes(os.urandom(8), byteorder="big", signed=False)
    return int(base_seed)

def _safe_run_name(name):
    clean = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in str(name)).strip("_")
    return clean or "parallel_run"


def _trajectory_dir(base_output_dir, run_name, itraj):
    return Path(base_output_dir) / _safe_run_name(run_name) / f"traj_{itraj:06d}"


def _scratch_dir(scratch_base_dir, run_name, itraj):
    return Path(scratch_base_dir) / _safe_run_name(run_name) / f"traj_{itraj:06d}"


def _write_status(path, result):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_status(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _update_progress_from_line(result, status_file, line, progress_queue=None):
    match = _PROGRESS_RE.search(line)
    if match is None:
        return False

    _apply_progress(result, _progress_from_match(match))
    result.last_update = _utc_now()
    _write_status(status_file, result)
    _publish_progress(progress_queue, result)
    return True


def _publish_progress(progress_queue, result):
    if progress_queue is not None:
        progress_queue.put(asdict(result))


def _progress_from_match(match):
    rcom = match.group("rcom")
    return {
        "step": int(match.group("step")),
        "time_fs": float(match.group("time_fs")),
        "energy_drift_kjmol": float(match.group("energy_drift")),
        "temperature_K": float(match.group("temperature")),
        "rcom_angstrom": float(rcom) if rcom is not None else None,
    }


def _apply_progress(target, progress):
    if isinstance(target, dict):
        target.update({key: value for key, value in progress.items() if value is not None})
        target["last_update"] = _utc_now()
        return

    target.step = progress["step"]
    target.time_fs = progress["time_fs"]
    target.energy_drift_kjmol = progress["energy_drift_kjmol"]
    target.temperature_K = progress["temperature_K"]
    if progress["rcom_angstrom"] is not None:
        target.rcom_angstrom = progress["rcom_angstrom"]


class _ParallelStdout:
    def __init__(self, result, status_file, initial_conditions_file, output_log_file, progress_queue=None):
        self.result = result
        self.status_file = status_file
        self.initial_conditions_file = initial_conditions_file
        self.output_log_file = output_log_file
        self.progress_queue = progress_queue
        self._buffer = ""
        self._saw_progress = False

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)

    def flush(self):
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""

    def _handle_line(self, line):
        if not line.strip():
            return

        if _update_progress_from_line(self.result, self.status_file, line, self.progress_queue):
            self._saw_progress = True
            return

        target = self.output_log_file if self._saw_progress else self.initial_conditions_file
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def _drain_progress_queue(progress_queue, progress_by_traj):
    if progress_queue is None:
        return False
    updated = False
    while True:
        try:
            progress = progress_queue.get_nowait()
        except queue_module.Empty:
            return updated
        progress_by_traj[progress["itraj"]] = progress
        updated = True


def _resolve_progress_mode(progress_report, progress_mode):
    if not progress_report:
        return "none"
    if progress_mode == "auto":
        return "table" if sys.stdout.isatty() else "log"
    if progress_mode not in {"table", "log", "none"}:
        raise ValueError("progress_mode must be one of: 'auto', 'table', 'log', 'none'")
    return progress_mode


def _progress_rows(active_itraj, base_output_dir, run_name, progress_by_traj=None):
    rows = []
    for itraj in sorted(active_itraj):
        data = None if progress_by_traj is None else progress_by_traj.get(itraj)
        if data is None:
            status_path = _trajectory_dir(base_output_dir, run_name, itraj) / "status.json"
            data = _read_status(status_path) or {"itraj": itraj, "status": "queued"}
        if data.get("status") in {"finished", "failed"}:
            continue
        rows.append(data)
    return rows


def _progress_table_lines(rows, total_trajectories, finished_count):
    lines = [
        f"Parallel trajectories: {finished_count}/{total_trajectories} finished",
        f"{'traj':>6} {'status':>10} {'step':>10} {'t[fs]':>12} "
        f"{'dE[kJ/mol]':>14} {'T[K]':>10} {'Rcom[A]':>10}",
    ]
    for row in rows:
        lines.append(
            f"{int(row.get('itraj', -1)):6d} "
            f"{str(row.get('status', 'queued')):>10} "
            f"{_format_progress_value(row.get('step'), 0):>10} "
            f"{_format_progress_value(row.get('time_fs'), 2):>12} "
            f"{_format_progress_value(row.get('energy_drift_kjmol'), 3):>14} "
            f"{_format_progress_value(row.get('temperature_K'), 1):>10} "
            f"{_format_progress_value(row.get('rcom_angstrom'), 2):>10}"
        )
    return lines


def _render_progress(active_itraj, base_output_dir, run_name, total_trajectories, finished_count,
                     progress_by_traj=None, progress_mode="auto", last_table_lines=0):
    rows = _progress_rows(active_itraj, base_output_dir, run_name, progress_by_traj=progress_by_traj)

    if progress_mode == "none":
        return 0

    if progress_mode == "log":
        if rows:
            latest = max(rows, key=lambda row: row.get("last_update") or "")
            sys.stdout.write(
                f"Parallel trajectories: {finished_count}/{total_trajectories} finished | "
                f"active={len(rows)} | latest traj={latest.get('itraj')} "
                f"step={_format_progress_value(latest.get('step'), 0)} "
                f"t[fs]={_format_progress_value(latest.get('time_fs'), 2)} "
                f"dE[kJ/mol]={_format_progress_value(latest.get('energy_drift_kjmol'), 3)} "
                f"T[K]={_format_progress_value(latest.get('temperature_K'), 1)} "
                f"Rcom[A]={_format_progress_value(latest.get('rcom_angstrom'), 2)}\n"
            )
        else:
            sys.stdout.write(f"Parallel trajectories: {finished_count}/{total_trajectories} finished | active=0\n")
        sys.stdout.flush()
        return 0

    lines = _progress_table_lines(rows, total_trajectories, finished_count)
    if last_table_lines > 0:
        sys.stdout.write(f"\033[{last_table_lines}F")
    for line in lines:
        sys.stdout.write("\033[2K" + line + "\n")
    for _ in range(max(0, last_table_lines - len(lines))):
        sys.stdout.write("\033[2K\n")
    sys.stdout.flush()
    return max(last_table_lines, len(lines))


def _format_progress_value(value, digits):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _write_initial_conditions_summary(system, context):
    with open(context.initial_conditions_file, "a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("Final sampled initial conditions\n")
        handle.write(f"trajectory_index {context.itraj}\n")
        handle.write(f"trajectory_dir {context.trajectory_dir}\n")
        handle.write(f"scratch_dir {context.scratch_dir}\n")
        handle.write(f"traj_file {context.traj_file}\n")
        handle.write(f"backfile {context.backfile}\n")
        handle.write(f"nproc_per_job {context.nproc_per_job}\n")

        qchem = getattr(system, "qchem", None)
        if isinstance(qchem, dict):
            handle.write(f"qchem {qchem.get('qchem')}\n")
            handle.write(f"qchem_nproc {qchem.get('nproc')}\n")
            handle.write(f"qchem_scratch_dir {qchem.get('scratch_dir')}\n")

        if hasattr(system, "Rini") and system.Rini is not None:
            handle.write(f"Rini[A] {system.Rini * BOHR_TO_ANGSTROM:.10f}\n")
        if hasattr(system, "bmax") and system.bmax is not None:
            handle.write(f"bmax[A] {system.bmax * BOHR_TO_ANGSTROM:.10f}\n")
        if hasattr(system, "bimp") and system.bimp is not None:
            handle.write(f"bimp[A] {system.bimp * BOHR_TO_ANGSTROM:.10f}\n")
        if hasattr(system, "Ecoll") and system.Ecoll is not None:
            handle.write(f"Ecoll[Eh] {system.Ecoll:.12f}\n")
        if hasattr(system, "tempcoll") and system.tempcoll is not None:
            handle.write(f"collision_temperature[K] {system.tempcoll:.6f}\n")
        sampling_metadata = getattr(system, "sampling_metadata", None)
        sampling_warnings = getattr(system, "sampling_warnings", None)
        sampling_records = list(sampling_metadata or [])
        if sampling_warnings is not sampling_metadata:
            sampling_records.extend(sampling_warnings or [])
        for record in sampling_records:
            prefix = "sampling_warning" if str(record.get("type", "")).startswith((
                "excluded_", "absolute_value_"
            )) else "sampling_metadata"
            handle.write(f"{prefix} {json.dumps(record, sort_keys=True)}\n")

        atoms = getattr(system, "atoms", None)
        q = getattr(system, "q", None)
        p = getattr(system, "p", None)
        if atoms is None or q is None or p is None:
            return

        handle.write("\n")
        handle.write("atom       x[A]             y[A]             z[A]             px[au]           py[au]           pz[au]\n")
        for i, atom in enumerate(atoms):
            j = 3 * i
            handle.write(
                f"{atom:>3s} "
                f"{q[j] * BOHR_TO_ANGSTROM:16.8f} "
                f"{q[j + 1] * BOHR_TO_ANGSTROM:16.8f} "
                f"{q[j + 2] * BOHR_TO_ANGSTROM:16.8f} "
                f"{p[j]:16.8f} "
                f"{p[j + 1]:16.8f} "
                f"{p[j + 2]:16.8f}\n"
            )


def _wrap_sampling_summary(system, context, method_name):
    if not hasattr(system, method_name):
        return

    original = getattr(system, method_name)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        _write_initial_conditions_summary(system, context)
        return result

    setattr(system, method_name, wrapped)


def _install_progress_callback(system, result, status_file, progress_queue):
    def progress_callback(**progress):
        _apply_progress(result, progress)
        result.status = "running"
        result.last_update = _utc_now()
        _write_status(status_file, result)
        _publish_progress(progress_queue, result)

    setattr(system, "_parallel_progress_callback", progress_callback)


def _is_pes_qchem(qchem):
    return isinstance(qchem, dict) and qchem.get("qchem") == "PES"


def _configured_qchem(qchem, scratch_dir, nproc_per_job, pes_single_core):
    if not isinstance(qchem, dict):
        return qchem

    configured = copy.deepcopy(qchem)
    configured["scratch_dir"] = str(scratch_dir)
    configured["nproc"] = int(nproc_per_job)
    configured = validate_qchem_input(configured)
    if pes_single_core and _is_pes_qchem(configured):
        configured["nproc"] = 1
    return configured


def _configure_system_runtime(system, scratch_dir, nproc_per_job, pes_single_core=True):
    seen = set()
    objects = [system]
    for attr in ("fragment_A", "fragment_B"):
        fragment = getattr(system, attr, None)
        if fragment is not None:
            objects.append(fragment)

    for obj in objects:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "qchem") and isinstance(obj.qchem, dict):
            obj.qchem = _configured_qchem(
                obj.qchem,
                scratch_dir=scratch_dir,
                nproc_per_job=nproc_per_job,
                pes_single_core=pes_single_core,
            )


def collision_single_run(system, context, **run_kwargs):
    traj_file = run_kwargs.pop("traj_file", context.traj_file)
    backfile = run_kwargs.pop("backfile", context.backfile)
    run_kwargs.setdefault("traj_index", context.itraj)
    run_kwargs.setdefault("mode_energy_file", str(Path(context.trajectory_dir) / "sampled_mode_energy_diagnostics.dat"))
    run_kwargs.setdefault("seed_metadata_file", str(Path(context.trajectory_dir) / "sampling_seed_metadata.jsonl"))
    _wrap_sampling_summary(system, context, "sample_bimolecular_reactants")
    return system.sample_and_run_collision(
        traj_file=traj_file,
        backfile=backfile,
        **run_kwargs,
    )


def unimolecular_single_run(system, context, **run_kwargs):
    traj_file = run_kwargs.pop("traj_file", context.traj_file)
    backfile = run_kwargs.pop("backfile", context.backfile)
    run_kwargs.setdefault("traj_index", context.itraj)
    run_kwargs.setdefault("mode_energy_file", str(Path(context.trajectory_dir) / "sampled_mode_energy_diagnostics.dat"))
    run_kwargs.setdefault("seed_metadata_file", str(Path(context.trajectory_dir) / "sampling_seed_metadata.jsonl"))
    _wrap_sampling_summary(system, context, "atom_sampling")
    _wrap_sampling_summary(system, context, "diatom_sampling")
    _wrap_sampling_summary(system, context, "polyatom_sampling")
    return system.sample_and_run_trajectory(
        traj_file=traj_file,
        backfile=backfile,
        **run_kwargs,
    )


def _run_one_from_factory(
    itraj,
    system_factory,
    single_run,
    trajectory_dir,
    scratch_dir,
    nproc_per_job,
    run_kwargs,
    pes_single_core,
    base_seed=None,
    progress_queue=None,
):
    trajectory_path = Path(trajectory_dir)
    scratch_path = Path(scratch_dir)
    trajectory_path.mkdir(parents=True, exist_ok=True)
    scratch_path.mkdir(parents=True, exist_ok=True)

    trajectory_seed = derive_trajectory_seed(base_seed, itraj)
    context = TrajectoryContext(
        itraj=itraj,
        trajectory_dir=str(trajectory_path),
        scratch_dir=str(scratch_path),
        traj_file=str(trajectory_path / "traj.xyz"),
        backfile=str(trajectory_path / "backup.xyz"),
        initial_conditions_file=str(trajectory_path / f"initial_conditions_for_traj_{itraj:06d}.dat"),
        nproc_per_job=int(nproc_per_job),
        sampling_seed=trajectory_seed,
    )

    status_file = trajectory_path / "status.json"
    initial_conditions_file = Path(context.initial_conditions_file)
    output_log_file = trajectory_path / "suppressed_output.log"
    error_log_file = trajectory_path / "stderr.log"
    initial_conditions_file.write_text("", encoding="utf-8")
    output_log_file.write_text("", encoding="utf-8")
    error_log_file.write_text("", encoding="utf-8")
    result = ParallelTrajectoryResult(
        itraj=itraj,
        status="running",
        trajectory_dir=context.trajectory_dir,
        traj_file=context.traj_file,
        backfile=context.backfile,
        start_time=_utc_now(),
    )
    _write_status(status_file, result)

    try:
        stdout_capture = _ParallelStdout(
            result=result,
            status_file=status_file,
            initial_conditions_file=initial_conditions_file,
            output_log_file=output_log_file,
            progress_queue=progress_queue,
        )
        with open(error_log_file, "a", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_handle):
                set_sampling_seed(
                    context.sampling_seed,
                    label=f"parallel trajectory {itraj}",
                    metadata_file=str(trajectory_path / "sampling_seed_metadata.jsonl"),
                    print_report=False,
                )
                system = system_factory(itraj)
                _configure_system_runtime(
                    system,
                    scratch_dir=context.scratch_dir,
                    nproc_per_job=context.nproc_per_job,
                    pes_single_core=pes_single_core,
                )
                _install_progress_callback(system, result, status_file, progress_queue)
                single_run(system, context, **run_kwargs)
            stdout_capture.flush()
        result.status = "finished"
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = traceback_module.format_exc()
    finally:
        result.end_time = _utc_now()
        _write_status(status_file, result)
        _publish_progress(progress_queue, result)

    return result


def _template_factory(template, _itraj):
    return copy.deepcopy(template)


def _run_one_from_template(
    itraj,
    template,
    single_run,
    trajectory_dir,
    scratch_dir,
    nproc_per_job,
    run_kwargs,
    pes_single_core,
    base_seed=None,
    progress_queue=None,
):
    return _run_one_from_factory(
        itraj=itraj,
        system_factory=lambda index: _template_factory(template, index),
        single_run=single_run,
        trajectory_dir=trajectory_dir,
        scratch_dir=scratch_dir,
        nproc_per_job=nproc_per_job,
        run_kwargs=run_kwargs,
        pes_single_core=pes_single_core,
        base_seed=base_seed,
        progress_queue=progress_queue,
    )


def _resolve_nproc_per_job(nproc_per_job=None, cores_per_trajectory=None):
    if nproc_per_job is None:
        nproc_per_job = 1 if cores_per_trajectory is None else cores_per_trajectory
    elif cores_per_trajectory is not None and int(nproc_per_job) != int(cores_per_trajectory):
        raise ValueError("Use either nproc_per_job or cores_per_trajectory, not conflicting values")

    nproc_per_job = int(nproc_per_job)
    if nproc_per_job < 1:
        raise ValueError("nproc_per_job must be >= 1")
    return nproc_per_job


def _resolve_nparallel_jobs(nparallel_jobs=None, max_parallel_jobs=None):
    if nparallel_jobs is None:
        return max_parallel_jobs
    if max_parallel_jobs is not None and int(nparallel_jobs) != int(max_parallel_jobs):
        raise ValueError("Use either nparallel_jobs or max_parallel_jobs, not conflicting values")
    nparallel_jobs = int(nparallel_jobs)
    if nparallel_jobs < 1:
        raise ValueError("nparallel_jobs must be >= 1")
    return nparallel_jobs


def _default_max_parallel_jobs(slurm, nproc_per_job):
    if slurm:
        total_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    else:
        total_cores = os.cpu_count() or 1
    return max(1, total_cores // int(nproc_per_job))


def _submit_next(
    executor,
    pending,
    next_itraj,
    total_trajectories,
    job_factory,
):
    if next_itraj >= total_trajectories:
        return next_itraj

    future = executor.submit(*job_factory(next_itraj))
    pending[future] = next_itraj
    return next_itraj + 1


def run_parallel_trajectories(
    system_factory: Callable[[int], Any],
    single_run: Callable[..., Any],
    *,
    total_trajectories: int,
    nparallel_jobs: int | None = None,
    max_parallel_jobs: int | None = None,
    nproc_per_job: int | None = None,
    cores_per_trajectory: int | None = None,
    run_name: str = "parallel_run",
    base_output_dir: str = ".",
    scratch_base_dir: str = "scratch",
    run_kwargs: dict[str, Any] | None = None,
    slurm: bool = False,
    stop_on_error: bool = False,
    pes_single_core: bool = True,
    base_seed: int | None = None,
    progress_report: bool = True,
    progress_mode: str = "auto",
    progress_interval: float = 2.0,
):
    if total_trajectories < 1:
        raise ValueError("total_trajectories must be >= 1")
    nproc_per_job = _resolve_nproc_per_job(nproc_per_job, cores_per_trajectory)
    max_parallel_jobs = _resolve_nparallel_jobs(nparallel_jobs, max_parallel_jobs)
    if max_parallel_jobs is None:
        max_parallel_jobs = _default_max_parallel_jobs(slurm, nproc_per_job)
    if max_parallel_jobs < 1:
        raise ValueError("max_parallel_jobs must be >= 1")
    progress_mode = _resolve_progress_mode(progress_report, progress_mode)

    base_seed = _resolve_base_seed(base_seed)
    run_kwargs = dict(run_kwargs or {})
    active_jobs = min(max_parallel_jobs, total_trajectories)
    results = []
    pending = {}
    progress_by_traj = {}
    progress_queue = None
    last_table_lines = 0
    output_base_dir = str(Path(base_output_dir))
    scratch_base_dir = str(Path(scratch_base_dir))

    def job_factory(itraj):
        return (
            _run_one_from_factory,
            itraj,
            system_factory,
            single_run,
            str(_trajectory_dir(output_base_dir, run_name, itraj)),
            str(_scratch_dir(scratch_base_dir, run_name, itraj)),
            nproc_per_job,
            run_kwargs,
            pes_single_core,
            base_seed,
            progress_queue,
        )

    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=active_jobs) as executor:
            next_itraj = 0
            for _ in range(active_jobs):
                next_itraj = _submit_next(executor, pending, next_itraj, total_trajectories, job_factory)
            if progress_mode != "none":
                last_table_lines = _render_progress(
                    active_itraj=pending.values(),
                    base_output_dir=output_base_dir,
                    run_name=run_name,
                    total_trajectories=total_trajectories,
                    finished_count=len(results),
                    progress_by_traj=progress_by_traj,
                    progress_mode=progress_mode,
                    last_table_lines=last_table_lines,
                )

            while pending:
                done, _ = wait(pending, timeout=progress_interval, return_when=FIRST_COMPLETED)
                changed = _drain_progress_queue(progress_queue, progress_by_traj)
                for future in done:
                    itraj = pending.pop(future)
                    result = future.result()
                    results.append(result)
                    progress_by_traj[result.itraj] = asdict(result)
                    changed = True

                    if result.status == "failed" and stop_on_error:
                        for pending_future in pending:
                            pending_future.cancel()
                        raise RuntimeError(f"Parallel trajectory {itraj} failed: {result.error}")

                    next_itraj = _submit_next(executor, pending, next_itraj, total_trajectories, job_factory)
                if _drain_progress_queue(progress_queue, progress_by_traj):
                    changed = True
                if progress_mode != "none" and changed:
                    last_table_lines = _render_progress(
                        active_itraj=pending.values(),
                        base_output_dir=output_base_dir,
                        run_name=run_name,
                        total_trajectories=total_trajectories,
                        finished_count=len(results),
                        progress_by_traj=progress_by_traj,
                        progress_mode=progress_mode,
                        last_table_lines=last_table_lines,
                    )

    if progress_mode != "none":
        _render_progress(
            active_itraj=[],
            base_output_dir=output_base_dir,
            run_name=run_name,
            total_trajectories=total_trajectories,
            finished_count=len(results),
            progress_by_traj=progress_by_traj,
            progress_mode=progress_mode,
            last_table_lines=last_table_lines,
        )
    results.sort(key=lambda item: item.itraj)
    return results


def _run_parallel_from_template(
    template,
    single_run,
    *,
    total_trajectories,
    nparallel_jobs=None,
    max_parallel_jobs=None,
    nproc_per_job=None,
    cores_per_trajectory=None,
    run_name="parallel_run",
    base_output_dir=".",
    scratch_base_dir="scratch",
    run_kwargs=None,
    slurm=False,
    stop_on_error=False,
    pes_single_core=True,
    base_seed=None,
    progress_report=True,
    progress_mode="auto",
    progress_interval=2.0,
):
    if total_trajectories < 1:
        raise ValueError("total_trajectories must be >= 1")
    nproc_per_job = _resolve_nproc_per_job(nproc_per_job, cores_per_trajectory)
    max_parallel_jobs = _resolve_nparallel_jobs(nparallel_jobs, max_parallel_jobs)
    if max_parallel_jobs is None:
        max_parallel_jobs = _default_max_parallel_jobs(slurm, nproc_per_job)
    if max_parallel_jobs < 1:
        raise ValueError("max_parallel_jobs must be >= 1")
    progress_mode = _resolve_progress_mode(progress_report, progress_mode)

    base_seed = _resolve_base_seed(base_seed)
    run_kwargs = dict(run_kwargs or {})
    active_jobs = min(max_parallel_jobs, total_trajectories)
    results = []
    pending = {}
    progress_by_traj = {}
    progress_queue = None
    last_table_lines = 0
    output_base_dir = str(Path(base_output_dir))
    scratch_base_dir = str(Path(scratch_base_dir))

    def job_factory(itraj):
        return (
            _run_one_from_template,
            itraj,
            template,
            single_run,
            str(_trajectory_dir(output_base_dir, run_name, itraj)),
            str(_scratch_dir(scratch_base_dir, run_name, itraj)),
            nproc_per_job,
            run_kwargs,
            pes_single_core,
            base_seed,
            progress_queue,
        )

    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=active_jobs) as executor:
            next_itraj = 0
            for _ in range(active_jobs):
                next_itraj = _submit_next(executor, pending, next_itraj, total_trajectories, job_factory)
            if progress_mode != "none":
                last_table_lines = _render_progress(
                    active_itraj=pending.values(),
                    base_output_dir=output_base_dir,
                    run_name=run_name,
                    total_trajectories=total_trajectories,
                    finished_count=len(results),
                    progress_by_traj=progress_by_traj,
                    progress_mode=progress_mode,
                    last_table_lines=last_table_lines,
                )

            while pending:
                done, _ = wait(pending, timeout=progress_interval, return_when=FIRST_COMPLETED)
                changed = _drain_progress_queue(progress_queue, progress_by_traj)
                for future in done:
                    itraj = pending.pop(future)
                    result = future.result()
                    results.append(result)
                    progress_by_traj[result.itraj] = asdict(result)
                    changed = True

                    if result.status == "failed" and stop_on_error:
                        for pending_future in pending:
                            pending_future.cancel()
                        raise RuntimeError(f"Parallel trajectory {itraj} failed: {result.error}")

                    next_itraj = _submit_next(executor, pending, next_itraj, total_trajectories, job_factory)
                if _drain_progress_queue(progress_queue, progress_by_traj):
                    changed = True
                if progress_mode != "none" and changed:
                    last_table_lines = _render_progress(
                        active_itraj=pending.values(),
                        base_output_dir=output_base_dir,
                        run_name=run_name,
                        total_trajectories=total_trajectories,
                        finished_count=len(results),
                        progress_by_traj=progress_by_traj,
                        progress_mode=progress_mode,
                        last_table_lines=last_table_lines,
                    )

    if progress_mode != "none":
        _render_progress(
            active_itraj=[],
            base_output_dir=output_base_dir,
            run_name=run_name,
            total_trajectories=total_trajectories,
            finished_count=len(results),
            progress_by_traj=progress_by_traj,
            progress_mode=progress_mode,
            last_table_lines=last_table_lines,
        )
    results.sort(key=lambda item: item.itraj)
    return results


def run_parallel_collisions(
    collision_factory: Callable[[int], Any] | None = None,
    *,
    collision_template=None,
    total_trajectories: int,
    nparallel_jobs: int | None = None,
    max_parallel_jobs: int | None = None,
    nproc_per_job: int | None = None,
    cores_per_trajectory: int | None = None,
    run_name: str = "collisions",
    base_output_dir: str = ".",
    scratch_base_dir: str = "scratch",
    slurm: bool = False,
    stop_on_error: bool = False,
    pes_single_core: bool = True,
    base_seed: int | None = None,
    progress_report: bool = True,
    progress_mode: str = "auto",
    progress_interval: float = 2.0,
    **collision_run_kwargs,
):
    if collision_factory is None and collision_template is None:
        raise ValueError("Either collision_factory or collision_template must be provided")
    if collision_factory is not None and collision_template is not None:
        raise ValueError("Provide only one of collision_factory or collision_template")

    if collision_template is not None:
        return _run_parallel_from_template(
            collision_template,
            collision_single_run,
            total_trajectories=total_trajectories,
            nparallel_jobs=nparallel_jobs,
            max_parallel_jobs=max_parallel_jobs,
            nproc_per_job=nproc_per_job,
            cores_per_trajectory=cores_per_trajectory,
            run_name=run_name,
            base_output_dir=base_output_dir,
            scratch_base_dir=scratch_base_dir,
            run_kwargs=collision_run_kwargs,
            slurm=slurm,
            stop_on_error=stop_on_error,
            pes_single_core=pes_single_core,
            base_seed=base_seed,
            progress_report=progress_report,
            progress_mode=progress_mode,
            progress_interval=progress_interval,
        )

    return run_parallel_trajectories(
        collision_factory,
        collision_single_run,
        total_trajectories=total_trajectories,
        nparallel_jobs=nparallel_jobs,
        max_parallel_jobs=max_parallel_jobs,
        nproc_per_job=nproc_per_job,
        cores_per_trajectory=cores_per_trajectory,
        run_name=run_name,
        base_output_dir=base_output_dir,
        scratch_base_dir=scratch_base_dir,
        run_kwargs=collision_run_kwargs,
        slurm=slurm,
        stop_on_error=stop_on_error,
        pes_single_core=pes_single_core,
        base_seed=base_seed,
        progress_report=progress_report,
        progress_mode=progress_mode,
        progress_interval=progress_interval,
    )


def run_parallel_unimolecular(
    molecule_factory: Callable[[int], Any] | None = None,
    *,
    molecule_template=None,
    total_trajectories: int,
    nparallel_jobs: int | None = None,
    max_parallel_jobs: int | None = None,
    nproc_per_job: int | None = None,
    cores_per_trajectory: int | None = None,
    run_name: str = "unimolecular",
    base_output_dir: str = ".",
    scratch_base_dir: str = "scratch",
    slurm: bool = False,
    stop_on_error: bool = False,
    pes_single_core: bool = True,
    base_seed: int | None = None,
    progress_report: bool = True,
    progress_mode: str = "auto",
    progress_interval: float = 2.0,
    **trajectory_run_kwargs,
):
    if molecule_factory is None and molecule_template is None:
        raise ValueError("Either molecule_factory or molecule_template must be provided")
    if molecule_factory is not None and molecule_template is not None:
        raise ValueError("Provide only one of molecule_factory or molecule_template")

    if molecule_template is not None:
        return _run_parallel_from_template(
            molecule_template,
            unimolecular_single_run,
            total_trajectories=total_trajectories,
            nparallel_jobs=nparallel_jobs,
            max_parallel_jobs=max_parallel_jobs,
            nproc_per_job=nproc_per_job,
            cores_per_trajectory=cores_per_trajectory,
            run_name=run_name,
            base_output_dir=base_output_dir,
            scratch_base_dir=scratch_base_dir,
            run_kwargs=trajectory_run_kwargs,
            slurm=slurm,
            stop_on_error=stop_on_error,
            pes_single_core=pes_single_core,
            base_seed=base_seed,
            progress_report=progress_report,
            progress_mode=progress_mode,
            progress_interval=progress_interval,
        )

    return run_parallel_trajectories(
        molecule_factory,
        unimolecular_single_run,
        total_trajectories=total_trajectories,
        nparallel_jobs=nparallel_jobs,
        max_parallel_jobs=max_parallel_jobs,
        nproc_per_job=nproc_per_job,
        cores_per_trajectory=cores_per_trajectory,
        run_name=run_name,
        base_output_dir=base_output_dir,
        scratch_base_dir=scratch_base_dir,
        run_kwargs=trajectory_run_kwargs,
        slurm=slurm,
        stop_on_error=stop_on_error,
        pes_single_core=pes_single_core,
        base_seed=base_seed,
        progress_report=progress_report,
        progress_mode=progress_mode,
        progress_interval=progress_interval,
    )
