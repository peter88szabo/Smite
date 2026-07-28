import os


def prepare_wavefunction_output(filename, backend_name):
    if not filename:
        raise ValueError(f"{backend_name} wfu=True requires a wavefunction output filename")

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    return filename


def require_wavefunction_file(path, backend_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{backend_name} did not produce expected wavefunction file: {path}")
