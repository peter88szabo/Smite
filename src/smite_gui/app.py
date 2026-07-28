"""Application entry point for the Smite GUI."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is required for the Smite GUI. Install it with:\n"
            '  pip install -e ".[gui]"'
        ) from exc

    from smite_gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Smite Script Generator")
    window = MainWindow()
    window.resize(1180, 780)
    window.show()
    return app.exec()
