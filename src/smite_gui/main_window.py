"""Main PySide6 window for generating Smite input scripts."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from smite_gui.script_generator import (
    INTEGRATORS,
    SUPPORTED_FRAGMENT_TYPES,
    SUPPORTED_QCHEM,
    THERMOSTATS,
    CollisionConfig,
    FragmentConfig,
    QChemConfig,
    RunConfig,
    OptimizerConfig,
    generate_collision_script,
    generate_optimizer_script,
    generate_unimolecular_script,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smite Script Generator")

        self.tabs = QTabWidget()
        self.script_preview = QPlainTextEdit()
        self.script_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.script_preview.setFont(QFont("Monospace", 10))

        self.unimol_page = UnimolecularPage(self.update_script)
        self.collision_page = CollisionPage(self.update_script)
        self.optimizer_page = OptimizerPage(self.update_script)
        self.tabs.addTab(self.unimol_page, "Unimolecular")
        self.tabs.addTab(self.collision_page, "Collision")
        self.tabs.addTab(self.optimizer_page, "Optimizer")
        self.tabs.currentChanged.connect(self.update_script)

        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_script)
        save_button = QPushButton("Save As...")
        save_button.clicked.connect(self.save_script)

        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Generated Python script"))
        preview_header.addStretch(1)
        preview_header.addWidget(copy_button)
        preview_header.addWidget(save_button)

        right = QVBoxLayout()
        right.addLayout(preview_header)
        right.addWidget(self.script_preview)

        root = QHBoxLayout()
        root.addWidget(self.tabs, 2)
        preview_widget = QWidget()
        preview_widget.setLayout(right)
        root.addWidget(preview_widget, 3)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._build_menu()
        self.update_script()

    def _build_menu(self) -> None:
        save_action = QAction("Save Script As...", self)
        save_action.triggered.connect(self.save_script)
        self.menuBar().addMenu("File").addAction(save_action)

    def update_script(self, *_args) -> None:
        try:
            if self.tabs.currentWidget() is self.unimol_page:
                script = self.unimol_page.generate_script()
            elif self.tabs.currentWidget() is self.collision_page:
                script = self.collision_page.generate_script()
            else:
                script = self.optimizer_page.generate_script()
            self.script_preview.setPlainText(script)
        except Exception as exc:
            self.script_preview.setPlainText(f"# Cannot generate script yet:\n# {exc}")

    def copy_script(self) -> None:
        self.script_preview.selectAll()
        self.script_preview.copy()
        cursor = self.script_preview.textCursor()
        cursor.clearSelection()
        self.script_preview.setTextCursor(cursor)

    def save_script(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Smite script", "smite_run.py", "Python files (*.py)")
        if not path:
            return
        Path(path).write_text(self.script_preview.toPlainText(), encoding="utf-8")
        QMessageBox.information(self, "Saved", f"Script saved to:\n{path}")


class UnimolecularPage(QWidget):
    def __init__(self, on_change) -> None:
        super().__init__()
        self.fragment_form = FragmentForm("mol", "Molecule", on_change)
        self.run_form = RunForm(on_change)
        self.seed = IntLineEdit("13349112", on_change)

        layout = QVBoxLayout()
        layout.addWidget(InfoBox("Generate a script for one Fragment using sample_and_run_trajectory()."))
        layout.addWidget(self.fragment_form)
        layout.addWidget(_group("Random seed", _form_rows([("seed", self.seed)])))
        layout.addWidget(self.run_form)
        layout.addStretch(1)
        self.setLayout(_scroll_layout(layout))

    def generate_script(self) -> str:
        seed = self.seed.value_or_none()
        return generate_unimolecular_script(self.fragment_form.config(), self.run_form.config(), seed=seed)


class CollisionPage(QWidget):
    def __init__(self, on_change) -> None:
        super().__init__()
        self.fragment_a = FragmentForm("frag_a", "Fragment A", on_change)
        self.fragment_b = FragmentForm("frag_b", "Fragment B", on_change)
        self.reaction_qchem = QChemForm("Reaction qchem", on_change)
        self.collision_form = CollisionForm(on_change)
        self.run_form = RunForm(on_change)
        self.seed = IntLineEdit("28220222", on_change)

        layout = QVBoxLayout()
        layout.addWidget(InfoBox("Generate a script for two Fragments combined into a Collision."))
        layout.addWidget(self.fragment_a)
        layout.addWidget(self.fragment_b)
        layout.addWidget(self.reaction_qchem)
        layout.addWidget(self.collision_form)
        layout.addWidget(_group("Random seed", _form_rows([("seed", self.seed)])))
        layout.addWidget(self.run_form)
        layout.addStretch(1)
        self.setLayout(_scroll_layout(layout))

    def generate_script(self) -> str:
        seed = self.seed.value_or_none()
        return generate_collision_script(
            self.fragment_a.config(),
            self.fragment_b.config(),
            self.reaction_qchem.config(),
            self.collision_form.config(),
            self.run_form.config(),
            seed=seed,
        )


class OptimizerPage(QWidget):
    def __init__(self, on_change) -> None:
        super().__init__()
        self.qchem = QChemForm("Optimizer qchem", on_change)
        self.atoms = LineEdit("C,H,H,H,H", on_change)
        self.xyz = QPlainTextEdit()
        self.xyz.setPlaceholderText("Paste XYZ coordinates without atom count/header.")
        self.xyz.setPlainText(
            "C  0.000000  0.000000  0.000000\n"
            "H  0.627579  0.627579  0.627579\n"
            "H -0.627579 -0.627579  0.627579\n"
            "H -0.627579  0.627579 -0.627579\n"
            "H  0.627579 -0.627579 -0.627579"
        )
        self.xyz.textChanged.connect(on_change)
        self.method = ComboBox(("BFGS", "SD"), on_change)
        self.backend_optimizer = ComboBox(("auto", "smite", "backend"), on_change)
        self.maxstep = IntLineEdit("100", on_change)
        self.energy_tol = FloatLineEdit("5.0e-5", on_change)
        self.max_step = FloatLineEdit("4.0e-3", on_change)
        self.rms_step = FloatLineEdit("2.5e-3", on_change)
        self.max_gradient = FloatLineEdit("7.0e-4", on_change)
        self.rms_gradient = FloatLineEdit("5.0e-4", on_change)
        self.trajectory_file = LineEdit("geomopt_traj.xyz", on_change)
        self.print_report = CheckBox("Print optimizer report", True, on_change)
        self.orca_what = ComboBox(("minimum", "ts"), on_change)

        layout = QVBoxLayout()
        layout.addWidget(InfoBox("Generate a script that calls optimizer.optimize_geometry()."))
        layout.addWidget(self.qchem)
        layout.addWidget(_group("Geometry", _form_rows([("Atoms", self.atoms)])))
        layout.addWidget(QLabel("XYZ coordinates"))
        layout.addWidget(self.xyz)
        layout.addWidget(
            _group(
                "Optimizer settings",
                _form_rows(
                    [
                        ("Method", self.method),
                        ("Backend optimizer", self.backend_optimizer),
                        ("Max steps", self.maxstep),
                        ("Energy tolerance", self.energy_tol),
                        ("Max step", self.max_step),
                        ("RMS step", self.rms_step),
                        ("Max gradient", self.max_gradient),
                        ("RMS gradient", self.rms_gradient),
                        ("Trajectory file", self.trajectory_file),
                        ("Print report", self.print_report),
                        ("ORCA target", self.orca_what),
                    ]
                ),
            )
        )
        layout.addStretch(1)
        self.setLayout(_scroll_layout(layout))

    def generate_script(self) -> str:
        return generate_optimizer_script(
            OptimizerConfig(
                atoms=self.atoms.text(),
                xyz=self.xyz.toPlainText(),
                qchem=self.qchem.config(),
                method=self.method.currentText(),
                backend_optimizer=self.backend_optimizer.currentText(),
                maxstep=self.maxstep.value(),
                energy_tol=self.energy_tol.value(),
                max_step=self.max_step.value(),
                rms_step=self.rms_step.value(),
                max_gradient=self.max_gradient.value(),
                rms_gradient=self.rms_gradient.value(),
                trajectory_file=self.trajectory_file.text(),
                print_report=self.print_report.isChecked(),
                orca_what=self.orca_what.currentText(),
            )
        )


class FragmentForm(QGroupBox):
    def __init__(self, variable: str, title: str, on_change) -> None:
        super().__init__(title)
        self.variable = variable
        self.on_change = on_change

        self.fname = LineEdit(title.replace(" ", "_"), on_change)
        self.fragment_type = ComboBox(SUPPORTED_FRAGMENT_TYPES, on_change)
        self.atoms = LineEdit("H,O", on_change)
        self.req = FloatLineEdit("1.0", on_change)
        self.omega = FloatLineEdit("1000.0", on_change)
        self.diatom_model = ComboBox(("harmonic", "morse"), on_change)
        self.beta = FloatLineEdit("1.0", on_change)
        self.de = FloatLineEdit("100.0", on_change)
        self.rigid = CheckBox("Rigid", False, on_change)
        self.random_rot = CheckBox("Random rotation", True, on_change)
        self.linear = CheckBox("Linear polyatom", False, on_change)
        self.xyz = QPlainTextEdit()
        self.xyz.setPlaceholderText("Paste XYZ coordinates without atom count/header.")
        self.xyz.setPlainText("O 0.000 0.000 0.000\nH 0.000 0.000 0.960")
        self.xyz.textChanged.connect(on_change)
        self.qchem = QChemForm("Fragment qchem", on_change)
        self.init_vib_type = ComboBox(("ZPE", "Temp", "Q", "E"), on_change)
        self.init_rot_type = ComboBox(("Jfix", "Temp"), on_change)
        self.temp = FloatLineEdit("300.0", on_change)
        self.nvib = IntLineEdit("0", on_change)
        self.jrot = IntLineEdit("0", on_change)

        rows = [
            ("Smite name", self.fname),
            ("Type", self.fragment_type),
            ("Atoms", self.atoms),
            ("req [Angstrom]", self.req),
            ("omega [cm-1]", self.omega),
            ("Diatom model", self.diatom_model),
            ("Morse beta [Angstrom^-1]", self.beta),
            ("Morse De [kJ/mol]", self.de),
            ("Options", _row_widget([self.rigid, self.random_rot, self.linear])),
            ("init_vib_type", self.init_vib_type),
            ("init_rot_type", self.init_rot_type),
            ("temp [K]", self.temp),
            ("nvib", self.nvib),
            ("jrot", self.jrot),
        ]
        layout = QVBoxLayout()
        layout.addLayout(_form_rows(rows))
        layout.addWidget(QLabel("XYZ coordinates"))
        layout.addWidget(self.xyz)
        layout.addWidget(self.qchem)
        self.setLayout(layout)

    def config(self) -> FragmentConfig:
        return FragmentConfig(
            variable=self.variable,
            fname=self.fname.text() or self.variable,
            fragment_type=self.fragment_type.currentText(),
            atoms=self.atoms.text(),
            xyz=self.xyz.toPlainText(),
            qchem=self.qchem.config(),
            req=self.req.value(),
            omega=self.omega.value(),
            diatom_model=self.diatom_model.currentText(),
            beta=self.beta.value(),
            de=self.de.value(),
            rigid=self.rigid.isChecked(),
            random_rot=self.random_rot.isChecked(),
            linear=self.linear.isChecked(),
            init_vib_type=self.init_vib_type.currentText(),
            init_rot_type=self.init_rot_type.currentText(),
            temp=self.temp.value(),
            nvib=self.nvib.value(),
            jrot=self.jrot.value(),
        )


class QChemForm(QGroupBox):
    def __init__(self, title: str, on_change) -> None:
        super().__init__(title)
        self.qchem = ComboBox(SUPPORTED_QCHEM, on_change)
        self.path = LineEdit("", on_change)
        self.nproc = IntLineEdit("1", on_change)
        self.functional = LineEdit("", on_change)
        self.basis = LineEdit("", on_change)
        self.charge = IntLineEdit("0", on_change)
        self.multiplicity = IntLineEdit("1", on_change)
        self.additional = LineEdit("", on_change)
        self.wfu = CheckBox("Write wavefunction files", False, on_change)
        self.scratch_dir = LineEdit("", on_change)
        self.pes_path = LineEdit("", on_change)

        self.setLayout(
            _form_rows(
                [
                    ("Backend", self.qchem),
                    ("Executable path", self.path),
                    ("nproc", self.nproc),
                    ("Functional", self.functional),
                    ("Basis", self.basis),
                    ("Charge", self.charge),
                    ("Multiplicity", self.multiplicity),
                    ("Additional args", self.additional),
                    ("Scratch dir", self.scratch_dir),
                    ("PES path", self.pes_path),
                    ("Wavefunction", self.wfu),
                ]
            )
        )

    def config(self) -> QChemConfig:
        return QChemConfig(
            qchem=self.qchem.currentText(),
            path=self.path.text(),
            nproc=self.nproc.value(),
            functional=self.functional.text(),
            basis=self.basis.text(),
            charge=self.charge.value(),
            multiplicity=self.multiplicity.value(),
            additional=self.additional.text(),
            wfu=self.wfu.isChecked(),
            scratch_dir=self.scratch_dir.text(),
            pes_path=self.pes_path.text(),
        )


class CollisionForm(QGroupBox):
    def __init__(self, on_change) -> None:
        super().__init__("Collision sampling")
        self.rini = FloatLineEdit("5.0", on_change)
        self.bmax = FloatLineEdit("5.0", on_change)
        self.bsampling = QSpinBox()
        self.bsampling.setRange(0, 2)
        self.bsampling.setValue(2)
        self.bsampling.valueChanged.connect(on_change)
        self.ecoll = FloatLineEdit("10.0", on_change)
        self.ecoll_thermal = CheckBox("Thermal collision energy", False, on_change)
        self.temp = FloatLineEdit("300.0", on_change)
        self.post_collision_analysis = CheckBox("Post-collision vector analysis", False, on_change)
        self.setLayout(
            _form_rows(
                [
                    ("Rini [Angstrom]", self.rini),
                    ("bmax [Angstrom]", self.bmax),
                    ("bsampling 0/1/2", self.bsampling),
                    ("Ecoll [kJ/mol]", self.ecoll),
                    ("Ecoll thermal", self.ecoll_thermal),
                    ("Temperature [K]", self.temp),
                    ("Analysis", self.post_collision_analysis),
                ]
            )
        )

    def config(self) -> CollisionConfig:
        return CollisionConfig(
            rini=self.rini.value(),
            bmax=self.bmax.value(),
            bsampling=self.bsampling.value(),
            ecoll=self.ecoll.value(),
            ecoll_thermal=self.ecoll_thermal.isChecked(),
            temp=self.temp.value(),
            post_collision_analysis=self.post_collision_analysis.isChecked(),
        )


class RunForm(QGroupBox):
    def __init__(self, on_change) -> None:
        super().__init__("Dynamics script settings")
        self.integrator = ComboBox(INTEGRATORS, on_change)
        self.integrator_order = IntLineEdit("4", on_change)
        self.timestep = FloatLineEdit("1.0", on_change)
        self.maxstep = IntLineEdit("1000", on_change)
        self.iprint = IntLineEdit("1", on_change)
        self.rstop = FloatLineEdit("10.0", on_change)
        self.spectrum = CheckBox("Collect spectrum history", False, on_change)
        self.thermostat = ComboBox(THERMOSTATS, on_change)
        self.thermo_param = FloatLineEdit("100.0", on_change)
        self.thermo_temp = FloatLineEdit("300.0", on_change)
        self.traj_file = LineEdit("", on_change)
        self.backfile = LineEdit("", on_change)
        self.setLayout(
            _form_rows(
                [
                    ("Integrator", self.integrator),
                    ("Integrator order", self.integrator_order),
                    ("Timestep [fs]", self.timestep),
                    ("Max steps", self.maxstep),
                    ("Print every", self.iprint),
                    ("Rstop [Angstrom]", self.rstop),
                    ("Spectrum", self.spectrum),
                    ("Thermostat", self.thermostat),
                    ("Thermostat parameter", self.thermo_param),
                    ("Thermostat temp [K]", self.thermo_temp),
                    ("Trajectory file", self.traj_file),
                    ("Backup file", self.backfile),
                ]
            )
        )

    def config(self) -> RunConfig:
        return RunConfig(
            integrator=self.integrator.currentText(),
            integrator_order=self.integrator_order.value(),
            timestep=self.timestep.value(),
            maxstep=self.maxstep.value(),
            iprint=self.iprint.value(),
            rstop=self.rstop.value(),
            spectrum=self.spectrum.isChecked(),
            thermostat=self.thermostat.currentText(),
            thermo_param=self.thermo_param.value(),
            thermo_temp=self.thermo_temp.value(),
            traj_file=self.traj_file.text(),
            backfile=self.backfile.text(),
        )


class InfoBox(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)


class LineEdit(QLineEdit):
    def __init__(self, text: str, on_change) -> None:
        super().__init__(text)
        self.textChanged.connect(on_change)


class FloatLineEdit(LineEdit):
    def value(self) -> float:
        return float(self.text() or 0.0)


class IntLineEdit(LineEdit):
    def value(self) -> int:
        return int(self.text() or 0)

    def value_or_none(self) -> int | None:
        text = self.text().strip()
        return int(text) if text else None


class ComboBox(QComboBox):
    def __init__(self, values, on_change) -> None:
        super().__init__()
        self.addItems(list(values))
        self.currentTextChanged.connect(on_change)


class CheckBox(QCheckBox):
    def __init__(self, text: str, checked: bool, on_change) -> None:
        super().__init__(text)
        self.setChecked(checked)
        self.toggled.connect(on_change)


def _form_rows(rows) -> QFormLayout:
    layout = QFormLayout()
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for label, widget in rows:
        layout.addRow(label, widget)
    return layout


def _row_widget(widgets) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    row.setLayout(layout)
    return row


def _group(title: str, layout: QFormLayout) -> QGroupBox:
    group = QGroupBox(title)
    group.setLayout(layout)
    return group


def _scroll_layout(layout: QVBoxLayout) -> QVBoxLayout:
    inner = QWidget()
    inner.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    outer = QVBoxLayout()
    outer.addWidget(scroll)
    return outer
