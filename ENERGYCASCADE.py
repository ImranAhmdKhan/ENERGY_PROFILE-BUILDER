import os
os.environ['QT_API'] = 'pyside6'

import sys
import re
import json
import csv
import copy
import numpy as np
import matplotlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from itertools import cycle
from io import BytesIO

matplotlib.use('QtAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import style as mpl_style

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFormLayout, QGroupBox, QFileDialog, QMessageBox, QColorDialog,
    QFrame, QScrollArea, QDialog, QDialogButtonBox,
    QAbstractItemView, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolBar, QMenu, QStyle, QProgressDialog, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QSize, QThread, QTimer
from PySide6.QtGui import (
    QIcon, QColor, QAction, QPixmap, QImageReader,
    QPainter, QBrush, QPen, QLinearGradient, QKeySequence
)

# ==================== Connection-line & fancy arrow helpers ====================
def _draw_wavy(ax, x1, y1, x2, y2, n_waves=5, amp_frac=0.06, **kw):
    t = np.linspace(0, 1, 300)
    dx, dy = x2-x1, y2-y1
    length = max(np.hypot(dx, dy), 1e-9)
    px, py = -dy/length, dx/length
    amp = amp_frac * length
    wave = amp * np.sin(n_waves * 2 * np.pi * t)
    ax.plot(x1+t*dx+wave*px, y1+t*dy+wave*py, **kw)

def _draw_zigzag(ax, x1, y1, x2, y2, n_zigs=7, amp_frac=0.05, **kw):
    n = n_zigs*2 + 1
    t = np.linspace(0, 1, n)
    dx, dy = x2-x1, y2-y1
    length = max(np.hypot(dx, dy), 1e-9)
    px, py = -dy/length, dx/length
    amp = amp_frac * length
    displace = np.array([amp*((-1)**i) if i%2==1 else 0 for i in range(n)])
    ax.plot(x1+t*dx+displace*px, y1+t*dy+displace*py, **kw)

def _draw_curled(ax, x1, y1, x2, y2, rad=0.35, **kw):
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    mx, my = (x1+x2)/2, (y1+y2)/2
    dx, dy = x2-x1, y2-y1
    length = max(np.hypot(dx, dy), 1e-9)
    px, py = -dy/length, dx/length
    cp = (mx + rad*length*px, my + rad*length*py)
    path = mpath.Path([(x1,y1), cp, (x2,y2)],
                      [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])
    lw    = kw.pop('lw', kw.pop('linewidth', 1.0))
    color = kw.pop('color', 'k')
    alpha = kw.pop('alpha', 1.0)
    ls    = kw.pop('ls', kw.pop('linestyle', '-'))
    zorder= kw.pop('zorder', 3)
    ax.add_patch(__import__('matplotlib.patches', fromlist=['PathPatch']).PathPatch(
        path, fill=False, edgecolor=color,
        linewidth=lw, alpha=alpha, linestyle=ls, zorder=zorder))

def _draw_fancy_arrow(ax, x1, y1, x2, y2, style='Straight', color='k',
                      lw=1.0, alpha=0.75, ls='-', zorder=10, head_size=0.012):
    """Draw a styled arrow (wavy / zigzag / curled / straight) with an arrowhead at (x2,y2)."""
    import matplotlib.patches as mpatches
    kw = dict(color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder)
    # Draw body (stop slightly before tip to leave room for arrowhead)
    dx, dy   = x2-x1, y2-y1
    length   = max(np.hypot(dx, dy), 1e-9)
    ux, uy   = dx/length, dy/length             # unit vector toward tip
    gap      = head_size * length               # shorten body by arrowhead size
    bx2, by2 = x2 - gap*ux, y2 - gap*uy        # body end point

    if style in ('Straight', 'Default', ''):
        ax.plot([x1, bx2], [y1, by2], **kw)
    elif style == 'Wavy':
        _draw_wavy(ax, x1, y1, bx2, by2, n_waves=5, amp_frac=0.06, **kw)
    elif style == 'Zigzag':
        _draw_zigzag(ax, x1, y1, bx2, by2, n_zigs=6, amp_frac=0.05, **kw)
    elif style in ('Curled (Arc)', 'Curled'):
        _draw_curled(ax, x1, y1, bx2, by2, rad=0.30, **kw)
    elif style in ('Smooth (Bezier)', 'Smooth'):
        xs = np.linspace(x1, bx2, 60)
        ts = (xs-x1) / max(bx2-x1, 1e-9)
        ax.plot(xs, y1+(by2-y1)*(3*ts**2-2*ts**3), **kw)
    else:
        ax.plot([x1, bx2], [y1, by2], **kw)

    # Arrowhead (FancyArrow at tip)
    head_w = head_size * length * 0.8
    head_l = gap
    ax.annotate('', xy=(x2, y2), xytext=(bx2, by2),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, alpha=alpha,
                                mutation_scale=10*lw),
                zorder=zorder+1)

# ==================== Custom Widgets ====================
class StyledListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QListWidget { background-color:#ffffff; border:1px solid #d1d5db;
                border-radius:8px; padding:4px; font-size:12px; }
            QListWidget::item { padding:8px; margin:2px; border-radius:6px; }
            QListWidget::item:selected { background-color:#3b82f6; color:white; }
            QListWidget::item:hover:!selected { background-color:#f3f4f6; }
        """)

class GradientButton(QPushButton):
    def __init__(self, text="", parent=None, color1="#3b82f6", color2="#2563eb"):
        super().__init__(text, parent)
        self.color1 = QColor(color1)
        self.color2 = QColor(color2)
        self.setMinimumHeight(38)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton { border:none; border-radius:8px; font-weight:bold;
                font-size:13px; padding:8px 16px; }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        alpha = 200 if self.isEnabled() else 80
        c1 = QColor(self.color1); c1.setAlpha(alpha)
        c2 = QColor(self.color2); c2.setAlpha(alpha)
        gradient.setColorAt(0, c1)
        gradient.setColorAt(1, c2)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)
        painter.setPen(QPen(Qt.white))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class ModernGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox { font-weight:bold; border:1px solid #e5e7eb; border-radius:12px;
                margin-top:12px; padding-top:16px; background-color:#ffffff; font-size:13px; }
            QGroupBox::title { subcontrol-origin:margin; left:16px; padding:0 8px; color:#1f2937; }
        """)

# ==================== Data Classes ====================
@dataclass
class PlotItem:
    file_data: List[Tuple[str, float]]
    display_name: str
    folder_path: str
    pathway: str = ""                   # ★ NEW: groups states into one connected cascade
    custom_name: Optional[str] = None
    custom_color: Optional[str] = None
    energy: Optional[float] = None
    label_offset: List[float] = field(default_factory=lambda: [0.0, 0.5])
    delta_offset: List[float] = field(default_factory=lambda: [0.05, 0.0])
    label_font_size: Optional[int] = None
    label_font_weight: str = "normal"
    label_font_color: Optional[str] = None
    # ★ Extra thermo display per state
    show_extra_thermo: bool = False
    extra_thermo_type: str = "Enthalpy"   # "Enthalpy", "Electronic", "Both"
    # ★ Per-state connection arrow style (overrides global)
    conn_arrow_style: str = "Default"     # "Default","Wavy","Zigzag","Curled","Straight","Smooth"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

# ==================== Label Editor Dialog ====================
class LabelEditorDialog(QDialog):
    def __init__(self, plot_item, all_pathways=None, parent=None):
        super().__init__(parent)
        self.plot_item = plot_item
        self.all_pathways = all_pathways or []
        self.setWindowTitle("Edit Label & Pathway Properties")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background-color:#f9fafb; }
            QGroupBox { font-weight:bold; border:1px solid #e5e7eb; border-radius:8px;
                margin-top:8px; padding-top:12px; }
            QLabel { font-size:12px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding:6px; border:1px solid #d1d5db; border-radius:6px; background:white; }
        """)
        layout = QVBoxLayout(self)

        # State label
        name_group = QGroupBox("State Label")
        name_layout = QFormLayout(name_group)
        self.name_edit = QLineEdit(plot_item.custom_name or plot_item.display_name)
        name_layout.addRow("Label Text:", self.name_edit)
        layout.addWidget(name_group)

        # Pathway assignment  ← NEW
        pw_group = QGroupBox("Pathway Assignment")
        pw_layout = QFormLayout(pw_group)
        self.pathway_combo = QComboBox()
        self.pathway_combo.setEditable(True)
        known = sorted(set(self.all_pathways))
        self.pathway_combo.addItems(known)
        current_pw = plot_item.pathway or ""
        idx = self.pathway_combo.findText(current_pw)
        if idx >= 0:
            self.pathway_combo.setCurrentIndex(idx)
        else:
            self.pathway_combo.setCurrentText(current_pw)
        pw_layout.addRow("Pathway:", self.pathway_combo)
        pw_layout.addRow(QLabel("<i>States sharing a pathway name are connected by lines.</i>"))
        layout.addWidget(pw_group)

        # Colors
        color_group = QGroupBox("Colors")
        color_layout = QFormLayout(color_group)
        self.state_color_btn = QPushButton("State Color")
        self.state_color_btn.clicked.connect(self.choose_state_color)
        self.state_color_btn.setStyleSheet(
            f"background-color:{plot_item.custom_color or '#3b82f6'};color:white;"
            "border-radius:6px;padding:6px;")
        self.label_color_btn = QPushButton("Label Color")
        self.label_color_btn.clicked.connect(self.choose_label_color)
        lc = plot_item.label_font_color or "#1f2937"
        self.label_color_btn.setStyleSheet(
            f"background-color:{lc};color:white;border-radius:6px;padding:6px;")
        color_layout.addRow("State Color:", self.state_color_btn)
        color_layout.addRow("Label Color:", self.label_color_btn)
        layout.addWidget(color_group)

        # Font
        font_group = QGroupBox("Font Properties")
        font_layout = QFormLayout(font_group)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(plot_item.label_font_size or 10)
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItems(["normal", "bold", "light"])
        self.font_weight_combo.setCurrentText(plot_item.label_font_weight)
        font_layout.addRow("Font Size:", self.font_size_spin)
        font_layout.addRow("Font Weight:", self.font_weight_combo)
        layout.addWidget(font_group)

        # Position (energy label)
        pos_group = QGroupBox("Energy Label Offset  (drag on plot or set here)")
        pos_layout = QFormLayout(pos_group)
        self.x_offset_spin = QDoubleSpinBox()
        self.x_offset_spin.setRange(-5.0, 5.0)
        self.x_offset_spin.setSingleStep(0.1)
        self.x_offset_spin.setValue(plot_item.label_offset[0])
        self.y_offset_spin = QDoubleSpinBox()
        self.y_offset_spin.setRange(-5.0, 5.0)
        self.y_offset_spin.setSingleStep(0.1)
        self.y_offset_spin.setValue(plot_item.label_offset[1])
        pos_layout.addRow("X Offset:", self.x_offset_spin)
        pos_layout.addRow("Y Offset:", self.y_offset_spin)
        layout.addWidget(pos_group)

        # Delta label offset
        delta_group = QGroupBox("Δ Label Offset  (drag on plot or set here)")
        delta_layout = QFormLayout(delta_group)
        self.dx_offset_spin = QDoubleSpinBox()
        self.dx_offset_spin.setRange(-5.0, 5.0)
        self.dx_offset_spin.setSingleStep(0.1)
        self.dx_offset_spin.setValue(plot_item.delta_offset[0])
        self.dy_offset_spin = QDoubleSpinBox()
        self.dy_offset_spin.setRange(-5.0, 5.0)
        self.dy_offset_spin.setSingleStep(0.1)
        self.dy_offset_spin.setValue(plot_item.delta_offset[1])
        delta_layout.addRow("X Offset:", self.dx_offset_spin)
        delta_layout.addRow("Y Offset:", self.dy_offset_spin)
        layout.addWidget(delta_group)

        # Extra thermo display  ★ NEW
        thermo_group = QGroupBox("Extra Thermodynamic Display")
        thermo_layout = QFormLayout(thermo_group)
        self.check_extra_thermo = QCheckBox("Show extra parameter below energy label")
        self.check_extra_thermo.setChecked(plot_item.show_extra_thermo)
        self.extra_thermo_combo = QComboBox()
        self.extra_thermo_combo.addItems(["Enthalpy", "Electronic", "Both"])
        self.extra_thermo_combo.setCurrentText(plot_item.extra_thermo_type)
        thermo_layout.addRow(self.check_extra_thermo)
        thermo_layout.addRow("Show:", self.extra_thermo_combo)
        layout.addWidget(thermo_group)

        # Per-state connection arrow style  ★ NEW
        arrow_group = QGroupBox("Connection Arrow Style (overrides global)")
        arrow_layout = QFormLayout(arrow_group)
        self.arrow_style_combo = QComboBox()
        self.arrow_style_combo.addItems(
            ["Default", "Straight", "Smooth (Bezier)",
             "Wavy", "Zigzag", "Curled (Arc)"])
        self.arrow_style_combo.setCurrentText(plot_item.conn_arrow_style)
        arrow_layout.addRow("Style:", self.arrow_style_combo)
        layout.addWidget(arrow_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset_to_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.state_color = plot_item.custom_color
        self.label_color = plot_item.label_font_color

    def choose_state_color(self):
        color = QColorDialog.getColor(QColor(self.state_color or "#3b82f6"))
        if color.isValid():
            self.state_color = color.name()
            self.state_color_btn.setStyleSheet(
                f"background-color:{self.state_color};color:white;border-radius:6px;padding:6px;")

    def choose_label_color(self):
        color = QColorDialog.getColor(QColor(self.label_color or "#1f2937"))
        if color.isValid():
            self.label_color = color.name()
            self.label_color_btn.setStyleSheet(
                f"background-color:{self.label_color};color:white;border-radius:6px;padding:6px;")

    def reset_to_defaults(self):
        self.name_edit.setText(self.plot_item.display_name)
        self.state_color = None
        self.label_color = None
        self.font_size_spin.setValue(10)
        self.font_weight_combo.setCurrentText("normal")
        self.x_offset_spin.setValue(0.0)
        self.y_offset_spin.setValue(0.5)
        self.dx_offset_spin.setValue(0.05)
        self.dy_offset_spin.setValue(0.0)
        self.check_extra_thermo.setChecked(False)
        self.extra_thermo_combo.setCurrentText("Enthalpy")
        self.arrow_style_combo.setCurrentText("Default")
        self.state_color_btn.setStyleSheet(
            "background-color:#3b82f6;color:white;border-radius:6px;padding:6px;")
        self.label_color_btn.setStyleSheet(
            "background-color:#1f2937;color:white;border-radius:6px;padding:6px;")

    def get_values(self):
        return {
            'custom_name':       self.name_edit.text().strip() or None,
            'pathway':           self.pathway_combo.currentText().strip(),
            'custom_color':      self.state_color,
            'label_font_color':  self.label_color,
            'label_font_size':   self.font_size_spin.value(),
            'label_font_weight': self.font_weight_combo.currentText(),
            'label_offset':      [self.x_offset_spin.value(), self.y_offset_spin.value()],
            'delta_offset':      [self.dx_offset_spin.value(), self.dy_offset_spin.value()],
            'show_extra_thermo': self.check_extra_thermo.isChecked(),
            'extra_thermo_type': self.extra_thermo_combo.currentText(),
            'conn_arrow_style':  self.arrow_style_combo.currentText(),
        }

# ==================== Global File Browser Dialog ====================
class GlobalFileBrowser(QDialog):
    def __init__(self, folders_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global File Explorer – All .out Files")
        self.resize(1000, 700)
        self.selected_files = []
        self.setStyleSheet("""
            QDialog { background-color:#f9fafb; }
            QLabel { font-size:12px; }
            QListWidget { border:1px solid #e5e7eb; border-radius:8px;
                background-color:white; font-size:12px; }
            QLineEdit { padding:8px; border:1px solid #d1d5db; border-radius:6px; font-size:12px; }
        """)
        layout = QVBoxLayout(self)
        header = QLabel("Global File Explorer")
        header.setStyleSheet("""
            QLabel { font-size:18px; font-weight:bold; padding:12px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3b82f6,stop:1 #1e40af);
                color:white; border-radius:8px; }
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        sl = QHBoxLayout()
        sl.addWidget(QLabel("🔍 Search:"))
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter by filename, subfolder, or parent folder…")
        self.search_bar.textChanged.connect(self.filter_list)
        sl.addWidget(self.search_bar)
        layout.addLayout(sl)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setAlternatingRowColors(True)
        layout.addWidget(self.file_list)

        self.status_label = QLabel("0 files found")
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel(
            "<i>💡 Ctrl / Shift to select multiple files.</i>"))

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Add Selected to Plot")
        btns.button(QDialogButtonBox.Ok).setStyleSheet(
            "background-color:#10b981;color:white;font-weight:bold;padding:8px 16px;border-radius:6px;")
        btns.accepted.connect(self.handle_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.all_items_data = []
        for path, data in folders_dict.items():
            folder_name = os.path.basename(path)
            for rel, full in data['files']:
                display = f"[{folder_name}] {rel}"
                self.all_items_data.append((display, path, full, data['color']))
        self.filter_list()

    def filter_list(self):
        self.file_list.clear()
        query = self.search_bar.text().lower()
        count = 0
        for display, fp, full, color in self.all_items_data:
            if query in display.lower():
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, (fp, full))
                item.setForeground(QColor(color))
                item.setToolTip(full)
                if count % 2 == 0:
                    item.setBackground(QColor(248, 250, 252))
                self.file_list.addItem(item)
                count += 1
        self.status_label.setText(
            f"{count} files found (out of {len(self.all_items_data)} total)")

    def handle_accept(self):
        for item in self.file_list.selectedItems():
            self.selected_files.append(item.data(Qt.UserRole))
        self.accept()

# ==================== Multipliers Dialog ====================
class MultipliersDialog(QDialog):
    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Multipliers for Summed State")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Configure Summed State</b>"))
        layout.addWidget(QLabel("<i>Define coefficients for linear combination of states:</i>"))

        self.inputs = []
        for f in files:
            name = os.path.basename(f)
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.01, 100.0)
            spin.setValue(1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            lbl = QLabel(f"× {name[:45]}{'…' if len(name) > 45 else ''}")
            lbl.setToolTip(f)
            rl.addWidget(spin)
            rl.addWidget(lbl)
            rl.addStretch()
            layout.addWidget(row)
            self.inputs.append((f, spin))

        layout.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Apply")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_values(self):
        return [(path, spin.value()) for path, spin in self.inputs]

# ==================== Parser & Cache ====================
_parser_cache: Dict[str, Any] = {}
_mtime_cache:  Dict[str, float] = {}


def _read_file_content(file_path: str) -> str:
    """Read a text file trying common encodings."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def _is_xtb_file(content: str) -> bool:
    """Return True when the content looks like an xTB output file."""
    # xTB output always contains the program banner "x T B" or "xtb"
    return bool(re.search(r'x\s*T\s*B|\bxtb\b', content, re.IGNORECASE))


def get_cached_parser(file_path: str):
    """Return a cached XtbParser or OrcaParser depending on the file content."""
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        return _make_parser(file_path)
    if file_path in _parser_cache and _mtime_cache.get(file_path) == mtime:
        return _parser_cache[file_path]
    parser = _make_parser(file_path)
    _parser_cache[file_path] = parser
    _mtime_cache[file_path] = mtime
    return parser


def _make_parser(file_path: str):
    content = _read_file_content(file_path)
    if _is_xtb_file(content):
        return XtbParser(file_path, content)
    return OrcaParser(file_path, content)


# ── xTB parser ──────────────────────────────────────────────────────────────
class XtbParser:
    """
    Parses xTB output files.

    Energy patterns (last occurrence is used — covers frequency / hessian runs
    where xTB prints a summary block at the very end):

      Gibbs (Total Free Energy):
        | TOTAL FREE ENERGY              -65.835028620044 Eh   |

      Enthalpy (Total Enthalpy):
        | TOTAL ENTHALPY                 -65.778637741497 Eh   |

      Electronic (Total Energy / SCF energy):
        | TOTAL ENERGY                   -67.029745257008 Eh   |
    """

    _PATTERNS = {
        'gibbs':      r'TOTAL\s+FREE\s+ENERGY\s+([-\d\.]+)\s+Eh',
        'enthalpy':   r'TOTAL\s+ENTHALPY\s+([-\d\.]+)\s+Eh',
        'electronic': r'TOTAL\s+ENERGY\s+([-\d\.]+)\s+Eh',
    }

    def __init__(self, file_path: str, content: str = ""):
        self.file_path = file_path
        self.filename  = os.path.basename(file_path)
        self.content   = content or _read_file_content(file_path)

    def get_energy(self, energy_type: str = "gibbs") -> Optional[float]:
        p = self._PATTERNS.get(energy_type.lower())
        if not p:
            return None
        m = re.findall(p, self.content)
        return float(m[-1]) if m else None

    def get_all_thermo(self) -> Dict[str, Optional[float]]:
        return {
            label: (
                (lambda hits: float(hits[-1]) if hits else None)
                (re.findall(self._PATTERNS[key], self.content))
            )
            for label, key in [
                ('Electronic', 'electronic'),
                ('Enthalpy',   'enthalpy'),
                ('Gibbs',      'gibbs'),
            ]
        }


# ── ORCA parser (unchanged, kept for ORCA .out files) ───────────────────────
class OrcaParser:
    def __init__(self, file_path: str, content: str = ""):
        self.file_path = file_path
        self.filename  = os.path.basename(file_path)
        self.content   = content or _read_file_content(file_path)

    def get_energy(self, energy_type: str = "gibbs") -> Optional[float]:
        patterns = {
            'gibbs':      r"Final Gibbs free energy\s*\.*\s*([-\d\.]+)",
            'enthalpy':   r"Total Enthalpy\s*\.*\s*([-\d\.]+)",
            'electronic': r"FINAL SINGLE POINT ENERGY\s+([-\d\.]+)",
        }
        p = patterns.get(energy_type.lower())
        if not p:
            return None
        m = re.findall(p, self.content)
        return float(m[-1]) if m else None

    def get_all_thermo(self) -> Dict[str, Optional[float]]:
        return {
            k: (lambda m: float(m[-1]) if m else None)(re.findall(p, self.content))
            for k, p in {
                'Electronic': r"FINAL SINGLE POINT ENERGY\s+([-\d\.]+)",
                'Enthalpy':   r"Total Enthalpy\s*\.*\s*([-\d\.]+)",
                'Gibbs':      r"Final Gibbs free energy\s*\.*\s*([-\d\.]+)",
            }.items()
        }

# ==================== Undo Manager ====================
class UndoManager:
    def __init__(self, max_depth=50):
        self._undo: List[Any] = []
        self._redo: List[Any] = []
        self._max = max_depth

    def push(self, lst):
        self._undo.append(copy.deepcopy(lst))
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current):
        if not self._undo:
            return None
        self._redo.append(copy.deepcopy(current))
        return self._undo.pop()

    def redo(self, current):
        if not self._redo:
            return None
        self._undo.append(copy.deepcopy(current))
        return self._redo.pop()

    def can_undo(self): return bool(self._undo)
    def can_redo(self): return bool(self._redo)

# ==================== Threading Worker ====================
class FolderLoader(QThread):
    finished   = Signal(dict)
    error      = Signal(str)
    progress   = Signal(int)
    files_found = Signal(int)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            all_files = []
            for root, _, filenames in os.walk(self.folder_path):
                if self._stop: return
                for f in filenames:
                    if f.lower().endswith('.out'):
                        all_files.append(os.path.join(root, f))
                        self.files_found.emit(len(all_files))

            if self._stop: return
            if not all_files:
                self.finished.emit({'path': self.folder_path, 'files': []})
                return

            files = []
            for i, fp in enumerate(all_files):
                if self._stop: return
                rel = os.path.relpath(fp, self.folder_path)
                files.append((rel, fp))
                self.progress.emit(i + 1)

            files.sort(key=lambda x: [
                int(c) if c.isdigit() else c.lower()
                for c in re.split(r'([0-9]+)', x[0])
            ])
            self.finished.emit({'path': self.folder_path, 'files': files})
        except Exception as e:
            self.error.emit(str(e))

# ==================== Constants ====================
_LINESTYLE_MAP = {
    'Solid (-)': '-', 'Dashed (--)': '--',
    'Dotted (:)': ':', 'Dash-dot (-.)': '-.',
}
_COLOR_CYCLE = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec489a', '#06b6d4', '#6b7280', '#84cc16', '#f97316',
]

# ==================== Main Application ====================
class OrcaAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORCA Professional Plotter v6.0 – Multi-Pathway Edition")
        self.resize(1850, 1060)
        QImageReader.setAllocationLimit(2048)

        self.folders: Dict[str, Dict] = {}
        self.plot_staging_list: List[PlotItem] = []
        self.pathway_colors: Dict[str, str] = {}   # ★ pathway name → hex color
        self.color_cycle = cycle(_COLOR_CYCLE)
        self.cross_diffs: List[Dict] = []          # ★ cross-pathway comparison specs

        self.dragging_idx = None
        self.dragging_type = None
        self._dragging_delta_key: Optional[Tuple[int, int]] = None
        self.pick_threshold = 0.30
        self.selected_states = set()
        self.undo_mgr = UndoManager()
        self._dirty = False
        self._active_loaders: List[FolderLoader] = []
        # Positions of rendered labels — updated every update_plot(), used for hit-testing
        self._label_plot_pos: Dict[int, Tuple[float, float]] = {}   # idx → (x, y)
        self._delta_plot_pos: Dict[Tuple[int,int], Tuple[float,float]] = {}  # (i,j) → (x, y)

        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.statusBar().showMessage(
            "Ready  |  v6.0: Multi-Pathway  |  1200 DPI Clipboard Active")
        # Show floating control panel beside the main window once Qt has positioned us
        QTimer.singleShot(150, self._show_float_ctrl)

    # ─────────────────────────────────────────────────────────────────
    # UI SETUP
    # ─────────────────────────────────────────────────────────────────
    def setup_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color:#f0f0f0; color:#111111;")
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setStyleSheet("""
            QSplitter::handle { background-color:#e5e7eb; border-radius:2px; }
            QSplitter::handle:hover { background-color:#9ca3af; }
        """)
        main_layout.addWidget(self.splitter)

        # ── Floating Control Window (lives outside the main GUI, freely resizable) ──
        self.float_ctrl_win = QWidget(
            None,
            Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint
        )
        self.float_ctrl_win.setWindowTitle("\u2699  ORCA Plotter \u2014 Controls")
        self.float_ctrl_win.resize(460, 900)
        self.float_ctrl_win.setMinimumWidth(380)
        self.float_ctrl_win.setMinimumHeight(400)
        self.float_ctrl_win.setStyleSheet("background:#f9fafb;")
        float_layout = QVBoxLayout(self.float_ctrl_win)
        float_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { border:none; background:transparent; }
            QScrollBar:vertical { border:none; background:#f3f4f6; width:12px; border-radius:6px; }
            QScrollBar::handle:vertical { background:#9ca3af; border-radius:6px; min-height:30px; }
            QScrollBar::handle:vertical:hover { background:#6b7280; }
        """)
        scroll_content = QWidget()
        self.sidebar_layout = QVBoxLayout(scroll_content)
        self.sidebar_layout.setSpacing(14)
        scroll.setWidget(scroll_content)
        float_layout.addWidget(scroll)

        # Header
        hdr = QLabel("ORCA Energy Plotter")
        hdr.setStyleSheet("""
            QLabel { font-size:20px; font-weight:bold; padding:12px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e3a8a,stop:1 #1e40af);
                color:#ffffff; border-radius:12px; }
        """)
        hdr.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(hdr)

        # ── Data Input ──
        inp_group = ModernGroupBox("📁 Data Input & Explorer")
        inp_lay = QVBoxLayout(inp_group)
        b_folder = GradientButton("Add Folder…", color1="#10b981", color2="#059669")
        b_folder.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        b_folder.clicked.connect(self.add_folder_dialog)
        b_browse = GradientButton("🌐 Global File Explorer", color1="#06b6d4", color2="#0891b2")
        b_browse.clicked.connect(self.open_global_browser)
        inp_lay.addWidget(b_folder)
        inp_lay.addWidget(b_browse)
        self.sidebar_layout.addWidget(inp_group)

        # Folder tabs
        self.tab_folders = QTabWidget()
        self.tab_folders.setStyleSheet("""
            QTabWidget::pane { border:1px solid #e5e7eb; border-radius:8px; background:white; }
            QTabBar::tab { background:#f3f4f6; border:1px solid #e5e7eb; padding:7px 14px;
                margin-right:3px; border-top-left-radius:6px; border-top-right-radius:6px; font-size:12px; }
            QTabBar::tab:selected { background:white; border-bottom-color:white; }
            QTabBar::tab:hover { background:#e5e7eb; }
        """)
        self.sidebar_layout.addWidget(self.tab_folders)

        # ── Pathway Manager ── ★ NEW section
        pw_group = ModernGroupBox("🔗 Pathway Manager")
        pw_lay = QVBoxLayout(pw_group)
        pw_lay.addWidget(QLabel(
            "<i>Group states into named cascades — even from the same folder.<br>"
            "States sharing a Pathway are drawn connected by lines.</i>"))

        pw_assign_lay = QHBoxLayout()
        self.pw_name_input = QLineEdit()
        self.pw_name_input.setPlaceholderText("Pathway name (e.g. 'Path A')…")
        self.pw_name_input.setStyleSheet(
            "padding:7px;border:1px solid #d1d5db;border-radius:6px;")
        btn_set_pw = QPushButton("Assign to Selected")
        btn_set_pw.setMinimumHeight(34)
        btn_set_pw.clicked.connect(self.set_pathway_for_selected)
        btn_pw_color = QPushButton("🎨 Pathway Color")
        btn_pw_color.setMinimumHeight(34)
        btn_pw_color.clicked.connect(self.choose_pathway_color)
        pw_assign_lay.addWidget(self.pw_name_input, 3)
        pw_assign_lay.addWidget(btn_set_pw, 2)
        pw_assign_lay.addWidget(btn_pw_color, 2)
        pw_lay.addLayout(pw_assign_lay)

        self.pw_list_widget = QListWidget()
        self.pw_list_widget.setMaximumHeight(80)
        self.pw_list_widget.setStyleSheet(
            "QListWidget { border:1px solid #e5e7eb; border-radius:6px; "
            "background:white; font-size:11px; }")
        pw_lay.addWidget(QLabel("Active pathways:"))
        pw_lay.addWidget(self.pw_list_widget)
        self.sidebar_layout.addWidget(pw_group)

        # ── Cross-Pathway Difference  ★ NEW ──
        xp_group = ModernGroupBox("↕ Cross-Pathway Differences")
        xp_lay = QVBoxLayout(xp_group)
        xp_lay.addWidget(QLabel(
            "<i>Highlight the energy gap between a state in one pathway\n"
            "and a state in another pathway.</i>"))

        xp_row1 = QHBoxLayout()
        xp_row1.addWidget(QLabel("State A:"))
        self.xp_state_a = QComboBox(); self.xp_state_a.setMinimumWidth(130)
        xp_row1.addWidget(self.xp_state_a, 1)
        xp_lay.addLayout(xp_row1)

        xp_row2 = QHBoxLayout()
        xp_row2.addWidget(QLabel("State B:"))
        self.xp_state_b = QComboBox(); self.xp_state_b.setMinimumWidth(130)
        xp_row2.addWidget(self.xp_state_b, 1)
        xp_lay.addLayout(xp_row2)

        xp_row3 = QHBoxLayout()
        xp_row3.addWidget(QLabel("Label:"))
        self.xp_label_input = QLineEdit()
        self.xp_label_input.setPlaceholderText("Auto")
        self.xp_label_input.setStyleSheet(
            "padding:5px;border:1px solid #d1d5db;border-radius:5px;")
        xp_row3.addWidget(self.xp_label_input, 1)
        xp_lay.addLayout(xp_row3)

        xp_row4 = QHBoxLayout()
        xp_row4.addWidget(QLabel("Arrow:"))
        self.xp_arrow_combo = QComboBox()
        self.xp_arrow_combo.addItems(
            ["Straight", "Wavy", "Zigzag", "Curled (Arc)", "Bracket"])
        xp_row4.addWidget(self.xp_arrow_combo, 1)
        xp_lay.addLayout(xp_row4)

        xp_btn_lay = QHBoxLayout()
        btn_xp_add = GradientButton("Add Comparison", color1="#f59e0b", color2="#d97706")
        btn_xp_add.setMinimumHeight(32)
        btn_xp_add.clicked.connect(self.add_cross_diff)
        btn_xp_clear = QPushButton("Clear All")
        btn_xp_clear.setMinimumHeight(32)
        btn_xp_clear.clicked.connect(self.clear_cross_diffs)
        xp_btn_lay.addWidget(btn_xp_add)
        xp_btn_lay.addWidget(btn_xp_clear)
        xp_lay.addLayout(xp_btn_lay)

        self.xp_list = QListWidget()
        self.xp_list.setMaximumHeight(90)
        self.xp_list.setStyleSheet(
            "QListWidget { border:1px solid #e5e7eb; border-radius:6px; "
            "background:white; font-size:11px; }")
        self.xp_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.xp_list.customContextMenuRequested.connect(self._xp_context_menu)
        xp_lay.addWidget(QLabel("Active comparisons (right-click to remove):"))
        xp_lay.addWidget(self.xp_list)
        self.sidebar_layout.addWidget(xp_group)

        # ── Staging ──
        stage_group = ModernGroupBox("📊 Plot Staging & Order")
        stage_lay = QVBoxLayout(stage_group)

        srch_lay = QHBoxLayout()
        self.staging_search = QLineEdit()
        self.staging_search.setPlaceholderText("Search states…")
        self.staging_search.textChanged.connect(self.filter_staging_list)
        srch_lay.addWidget(QLabel("🔍"))
        srch_lay.addWidget(self.staging_search)
        stage_lay.addLayout(srch_lay)

        self.staging_list = StyledListWidget()
        self.staging_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.staging_list.itemSelectionChanged.connect(self.on_staging_selection_changed)
        self.staging_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.staging_list.customContextMenuRequested.connect(self.show_staging_context_menu)
        self.staging_list.itemDoubleClicked.connect(lambda: self.open_label_editor())
        stage_lay.addWidget(self.staging_list)

        edit_lay = QHBoxLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Quick rename…")
        self.rename_input.setStyleSheet(
            "padding:7px;border:1px solid #d1d5db;border-radius:6px;")
        btn_rename = QPushButton("Rename")
        btn_rename.clicked.connect(self.rename_selected_state)
        btn_color  = QPushButton("Color")
        btn_color.clicked.connect(self.color_selected_state)
        btn_adv    = QPushButton("Advanced…")
        btn_adv.clicked.connect(self.open_label_editor)
        edit_lay.addWidget(self.rename_input)
        edit_lay.addWidget(btn_rename)
        edit_lay.addWidget(btn_color)
        edit_lay.addWidget(btn_adv)
        stage_lay.addLayout(edit_lay)

        stage_btns = QHBoxLayout()
        for label, fn in [
            ("↑ Up",    lambda: self.move_staging_item(-1)),
            ("↓ Down",  lambda: self.move_staging_item(1)),
            ("Remove",  self.remove_staged_items),
            ("Clear",   self.clear_staging),
        ]:
            b = QPushButton(label)
            b.setMinimumHeight(34)
            b.clicked.connect(fn)
            stage_btns.addWidget(b)
        stage_lay.addLayout(stage_btns)

        act_lay = QHBoxLayout()
        btn_plot = GradientButton("📈 Update Profile", color1="#10b981", color2="#059669")
        btn_plot.clicked.connect(self.update_plot)
        btn_clip = GradientButton("📋 Copy 1200 DPI", color1="#6b7280", color2="#4b5563")
        btn_clip.clicked.connect(self.copy_to_clipboard)
        act_lay.addWidget(btn_plot)
        act_lay.addWidget(btn_clip)
        stage_lay.addLayout(act_lay)
        self.sidebar_layout.addWidget(stage_group)

        # ── Journal Presets ──
        journal_group = ModernGroupBox("🎯 Publication Quality Presets")
        jl = QVBoxLayout(journal_group)
        self.journal_preset_combo = QComboBox()
        self.journal_preset_combo.addItems([
            "Custom", "JACS (Single Column)", "JACS (Double Column)",
            "Nature/Science", "Angewandte", "ACS Catalysis",
        ])
        self.journal_preset_combo.currentIndexChanged.connect(self.apply_journal_preset)
        jl.addWidget(self.journal_preset_combo)
        style_btns = QHBoxLayout()
        for name, (c1, c2) in [
            ("Light",    ("#8b5cf6", "#6d28d9")),
            ("Dark",     ("#1f2937", "#111827")),
            ("Colorful", ("#ec489a", "#db2777")),
        ]:
            b = QPushButton(name)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c1};
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 8px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {c2};
                }}
            """)
            b.clicked.connect(lambda checked, s=name: self.apply_quick_style(s))
            style_btns.addWidget(b)
        jl.addLayout(style_btns)
        self.sidebar_layout.addWidget(journal_group)

        # ── Legend & Labelling ──
        leg_group = ModernGroupBox("🏷️ Legend & Labeling")
        leg_form = QFormLayout(leg_group)
        leg_form.setVerticalSpacing(9)

        inter_lay = QHBoxLayout()
        self.check_interactive = QCheckBox("Enable Dragging")
        self.check_interactive.setChecked(True)
        self.btn_reset_labels = QPushButton("Reset Positions")
        self.btn_reset_labels.clicked.connect(self.reset_all_labels)
        inter_lay.addWidget(self.check_interactive)
        inter_lay.addWidget(self.btn_reset_labels)
        leg_form.addRow(inter_lay)

        self.check_show_legend  = QCheckBox("Display Legend")
        self.check_show_legend.setChecked(True)
        self.theme_combo        = QComboBox()
        self.theme_combo.addItems(["Default","ggplot","seaborn-v0_8","bmh",
                                    "dark_background","Solarize_Light2"])
        self.legend_loc_combo   = QComboBox()
        self.legend_loc_combo.addItems(["best","upper right","upper left",
                                         "lower left","lower right","right",
                                         "center left","center"])
        self.line_style_combo   = QComboBox()
        self.line_style_combo.addItems(["Solid (-)","Dashed (--)","Dotted (:)","Dash-dot (-.)"])

        # ── Global connection style (wavy / zigzag / curled…)  ★ NEW ──
        self.conn_style_combo = QComboBox()
        self.conn_style_combo.addItems([
            "Smooth (Bezier)", "Straight", "Wavy", "Zigzag", "Curled (Arc)"])
        self.conn_style_combo.setToolTip(
            "Shape of lines connecting adjacent states.\n"
            "Individual states can override this in Advanced editor.")
        self.conn_style_combo.currentIndexChanged.connect(self.update_plot)

        # ── Global extra thermo display  ★ NEW ──
        self.extra_thermo_display_combo = QComboBox()
        self.extra_thermo_display_combo.addItems(
            ["None", "Enthalpy", "Electronic", "Both"])
        self.extra_thermo_display_combo.setToolTip(
            "Show additional thermodynamic value(s) beneath the energy label.\n"
            "Per-state override available in Advanced editor.")
        self.extra_thermo_display_combo.currentIndexChanged.connect(self.update_plot)

        # ── Label mode & rotation ──
        self.label_mode_combo = QComboBox()
        self.label_mode_combo.addItems([
            "Tick Labels Only",
            "Inline Below Bar",
            "Inline Above Bar",
            "Both (Tick + Inline)",
            "No State Labels",
        ])
        self.label_mode_combo.setToolTip("Controls where state names appear on the plot.")
        self.label_mode_combo.currentIndexChanged.connect(self.update_plot)

        self.xlabel_rotation_spin = QSpinBox()
        self.xlabel_rotation_spin.setRange(0, 90)
        self.xlabel_rotation_spin.setValue(0)
        self.xlabel_rotation_spin.setSingleStep(15)
        self.xlabel_rotation_spin.setSuffix("°")
        self.xlabel_rotation_spin.setToolTip(
            "Rotation for x-axis tick labels (0=horizontal, 45=diagonal, 90=vertical).")
        self.xlabel_rotation_spin.valueChanged.connect(self.update_plot)

        self.check_smart_labels = QCheckBox("Smart Overlap Avoidance")
        self.check_smart_labels.setChecked(True)
        self.check_smart_labels.setToolTip(
            "Automatically shift energy value labels to avoid overlapping each other.")
        self.check_smart_labels.stateChanged.connect(self.update_plot)

        font_group = QGroupBox("Font Settings")
        ff = QFormLayout(font_group)
        self.base_fs_spin   = QSpinBox(); self.base_fs_spin.setRange(4,30); self.base_fs_spin.setValue(10)
        self.tick_fs_spin   = QSpinBox(); self.tick_fs_spin.setRange(4,30); self.tick_fs_spin.setValue(10)
        self.check_tick_bold= QCheckBox("Bold State Names")
        self.energy_fs_spin = QSpinBox(); self.energy_fs_spin.setRange(4,30); self.energy_fs_spin.setValue(9)
        self.check_energy_bold= QCheckBox("Bold Energy Labels"); self.check_energy_bold.setChecked(True)
        self.delta_fs_spin  = QSpinBox(); self.delta_fs_spin.setRange(4,30); self.delta_fs_spin.setValue(8)
        self.check_delta_bold = QCheckBox("Bold Delta Labels"); self.check_delta_bold.setChecked(True)
        ff.addRow("Base Size:", self.base_fs_spin)
        ff.addRow("Tick Size:", self.tick_fs_spin)
        ff.addRow("", self.check_tick_bold)
        ff.addRow("Energy Size:", self.energy_fs_spin)
        ff.addRow("", self.check_energy_bold)
        ff.addRow("Delta Size:", self.delta_fs_spin)
        ff.addRow("", self.check_delta_bold)

        self.check_pub_quality  = QCheckBox("Publication Style (Serif Font)")
        self.check_show_e_labels= QCheckBox("Show Energy Values"); self.check_show_e_labels.setChecked(True)
        self.check_show_delta   = QCheckBox("Show Δ Values");      self.check_show_delta.setChecked(True)
        self.check_auto_align   = QCheckBox("Auto-align Labels");  self.check_auto_align.setChecked(True)

        # ── Label / Delta box style  ★ NEW ──
        self.label_box_combo = QComboBox()
        self.label_box_combo.addItems(["Rounded Box", "Square Box", "No Box"])
        self.label_box_combo.setToolTip(
            "Box drawn around energy value labels.\n'No Box' = plain text only.")
        self.label_box_combo.currentIndexChanged.connect(self.update_plot)

        self.delta_box_combo = QComboBox()
        self.delta_box_combo.addItems(["Rounded Box", "Square Box", "No Box"])
        self.delta_box_combo.setToolTip("Box drawn around Δ value labels.")
        self.delta_box_combo.currentIndexChanged.connect(self.update_plot)

        # ── Delta arrow style  ★ NEW ──
        self.delta_arrow_combo = QComboBox()
        self.delta_arrow_combo.addItems(
            ["Straight →", "Wavy", "Zigzag", "Curled (Arc)", "No Arrow"])
        self.delta_arrow_combo.setToolTip(
            "Style of the arrow from the Δ label to the midpoint of the transition.\n"
            "Uses the same wavy/zigzag/curled engines as connection lines.")
        self.delta_arrow_combo.currentIndexChanged.connect(self.update_plot)

        leg_form.addRow(self.check_show_legend)
        leg_form.addRow("Theme:", self.theme_combo)
        leg_form.addRow("Legend Pos:", self.legend_loc_combo)
        leg_form.addRow("Line Style:", self.line_style_combo)
        leg_form.addRow("Connection:", self.conn_style_combo)
        leg_form.addRow("Δ Arrow Style:", self.delta_arrow_combo)
        leg_form.addRow("Extra Thermo:", self.extra_thermo_display_combo)
        leg_form.addRow("State Names:", self.label_mode_combo)
        leg_form.addRow("Tick Rotation:", self.xlabel_rotation_spin)
        leg_form.addRow(self.check_smart_labels)
        leg_form.addRow(font_group)
        leg_form.addRow(self.check_pub_quality)
        leg_form.addRow(self.check_show_e_labels)
        leg_form.addRow("Energy Label Box:", self.label_box_combo)
        leg_form.addRow(self.check_show_delta)
        leg_form.addRow("Δ Label Box:", self.delta_box_combo)
        leg_form.addRow(self.check_auto_align)

        for w in [self.check_show_legend, self.check_tick_bold,
                  self.check_energy_bold, self.check_delta_bold,
                  self.check_pub_quality, self.check_show_e_labels,
                  self.check_show_delta, self.check_interactive,
                  self.check_auto_align]:
            w.stateChanged.connect(self.update_plot)
        for w in [self.theme_combo, self.legend_loc_combo,
                  self.line_style_combo, self.conn_style_combo,
                  self.extra_thermo_display_combo,
                  self.label_box_combo, self.delta_box_combo,
                  self.delta_arrow_combo]:
            w.currentIndexChanged.connect(self.update_plot)
        for w in [self.base_fs_spin, self.tick_fs_spin,
                  self.energy_fs_spin, self.delta_fs_spin]:
            w.valueChanged.connect(self.update_plot)
        self.sidebar_layout.addWidget(leg_group)

        # ── Layout & Canvas ──
        box_group = ModernGroupBox("📐 Layout & Canvas")
        box_form = QFormLayout(box_group)
        box_form.setVerticalSpacing(9)

        self.fig_width_spin   = QDoubleSpinBox(); self.fig_width_spin.setRange(2,25); self.fig_width_spin.setValue(10); self.fig_width_spin.setSingleStep(0.5)
        self.fig_height_spin  = QDoubleSpinBox(); self.fig_height_spin.setRange(2,20); self.fig_height_spin.setValue(7); self.fig_height_spin.setSingleStep(0.5)
        self.state_width_slider= QDoubleSpinBox(); self.state_width_slider.setRange(0.1,0.9); self.state_width_slider.setValue(0.6); self.state_width_slider.setSingleStep(0.05)
        self.bar_thickness_spin= QDoubleSpinBox(); self.bar_thickness_spin.setRange(0.5,15); self.bar_thickness_spin.setValue(3.5); self.bar_thickness_spin.setSingleStep(0.5)
        self.margin_left_spin  = QDoubleSpinBox(); self.margin_left_spin.setRange(0.01,0.4); self.margin_left_spin.setValue(0.12); self.margin_left_spin.setSingleStep(0.01)
        self.pathway_gap_spin  = QDoubleSpinBox(); self.pathway_gap_spin.setRange(0,3); self.pathway_gap_spin.setValue(0.5); self.pathway_gap_spin.setSingleStep(0.25)  # ★ NEW

        vis1 = QHBoxLayout(); vis2 = QHBoxLayout()
        self.check_show_xaxis= QCheckBox("X-Axis"); self.check_show_xaxis.setChecked(True)
        self.check_show_yaxis= QCheckBox("Y-Axis"); self.check_show_yaxis.setChecked(True)
        self.check_show_frame= QCheckBox("Frame");  self.check_show_frame.setChecked(True)
        self.check_inward_ticks=QCheckBox("Inward Ticks"); self.check_inward_ticks.setChecked(True)
        self.check_grid  = QCheckBox("Grid");   self.check_grid.setChecked(True)
        self.check_smooth= QCheckBox("Smooth"); self.check_smooth.setChecked(True)
        self.check_overlay= QCheckBox("Overlay"); self.check_overlay.setChecked(True)
        self.check_shadow= QCheckBox("Shadows")
        vis1.addWidget(self.check_show_xaxis); vis1.addWidget(self.check_show_yaxis); vis1.addWidget(self.check_show_frame)
        vis2.addWidget(self.check_inward_ticks); vis2.addWidget(self.check_grid)
        vis2.addWidget(self.check_smooth); vis2.addWidget(self.check_overlay); vis2.addWidget(self.check_shadow)

        box_form.addRow("Width / Height (in):", self.fig_width_spin)
        box_form.addRow("", self.fig_height_spin)
        box_form.addRow("State Bar Width:", self.state_width_slider)
        box_form.addRow("Bar Thickness (pt):", self.bar_thickness_spin)
        box_form.addRow("Left Margin:", self.margin_left_spin)
        box_form.addRow("Pathway Gap:", self.pathway_gap_spin)   # ★ NEW
        box_form.addRow(vis1)
        box_form.addRow(vis2)

        for w in [self.check_show_xaxis, self.check_show_yaxis,
                  self.check_show_frame, self.check_inward_ticks,
                  self.check_grid, self.check_smooth,
                  self.check_overlay, self.check_shadow]:
            w.stateChanged.connect(self.update_plot)
        for w in [self.fig_width_spin, self.fig_height_spin,
                  self.state_width_slider, self.bar_thickness_spin,
                  self.margin_left_spin, self.pathway_gap_spin]:
            w.valueChanged.connect(self.update_plot)
        self.sidebar_layout.addWidget(box_group)

        # ── Axis & Reference ──
        cfg_group = ModernGroupBox("📊 Axis & Reference")
        cfg_form = QFormLayout(cfg_group)
        cfg_form.setVerticalSpacing(9)
        self.title_input  = QLineEdit("Reaction Energy Profile")
        self.title_input.setStyleSheet("padding:8px;border:1px solid #d1d5db;border-radius:6px;")
        self.y_label_input= QLineEdit("Relative Energy (kcal/mol)")
        self.y_label_input.setStyleSheet("padding:8px;border:1px solid #d1d5db;border-radius:6px;")
        self.energy_type_combo = QComboBox()
        self.energy_type_combo.addItems(["Gibbs","Enthalpy","Electronic"])
        self.zero_ref_combo = QComboBox()
        self.zero_ref_combo.addItem("Minimum (Global)")
        self.zero_ref_combo.addItem("Per Pathway (First State)")
        self.zero_ref_combo.currentIndexChanged.connect(self.update_plot)
        unit_lay = QHBoxLayout()
        self.btn_kcal = QPushButton("kcal/mol"); self.btn_kj = QPushButton("kJ/mol"); self.btn_ev = QPushButton("eV")
        self.btn_kcal.setCheckable(True); self.btn_kj.setCheckable(True); self.btn_ev.setCheckable(True)
        self.btn_kcal.setChecked(True)
        for b in [self.btn_kcal, self.btn_kj, self.btn_ev]:
            b.setMinimumHeight(30); b.clicked.connect(self.update_energy_units); unit_lay.addWidget(b)
        cfg_form.addRow("Title:", self.title_input)
        cfg_form.addRow("Y-Axis Label:", self.y_label_input)
        cfg_form.addRow("Energy Source:", self.energy_type_combo)
        cfg_form.addRow("Zero Reference:", self.zero_ref_combo)
        cfg_form.addRow("Units:", unit_lay)
        self.title_input.textChanged.connect(self.update_plot)
        self.y_label_input.textChanged.connect(self.update_plot)
        self.energy_type_combo.currentIndexChanged.connect(self.update_plot)
        self.sidebar_layout.addWidget(cfg_group)
        self.sidebar_layout.addStretch()

        # ── Main view (plot + table tabs) ──
        self.main_view_tabs = QTabWidget()
        self.main_view_tabs.setStyleSheet("""
            QTabWidget::pane { border:2px solid #e5e7eb; border-radius:12px;
                background:white; padding:8px; }
            QTabBar::tab { background:#f3f4f6; border:1px solid #e5e7eb;
                padding:10px 20px; margin-right:4px;
                border-top-left-radius:8px; border-top-right-radius:8px;
                font-weight:bold; font-size:13px; }
            QTabBar::tab:selected { background:white; border-bottom-color:white; color:#3b82f6; }
            QTabBar::tab:hover { background:#e5e7eb; }
        """)

        # Plot tab
        self.plot_tab = QWidget()
        ptl = QVBoxLayout(self.plot_tab)
        ptl.setContentsMargins(8, 8, 8, 8)
        ph = QLabel("Publication-Ready Energy Profile")
        ph.setStyleSheet("font-size:16px;font-weight:bold;color:#000;padding:8px;")
        ph.setAlignment(Qt.AlignCenter)
        ptl.addWidget(ph)
        cf = QFrame()
        cf.setFrameStyle(QFrame.Box | QFrame.Raised)
        cf.setStyleSheet("border:2px solid #e5e7eb;border-radius:8px;background:white;")
        cfl = QVBoxLayout(cf)
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, None)
        cfl.addWidget(self.canvas)
        cfl.addWidget(self.toolbar)
        ptl.addWidget(cf)
        self.main_view_tabs.addTab(self.plot_tab, "📈 Publication Plot")

        # Table tab
        self.data_tab = QWidget()
        dtl = QVBoxLayout(self.data_tab)
        dtl.setContentsMargins(8, 8, 8, 8)
        dh = QLabel("Thermodynamic Data")
        dh.setStyleSheet("font-size:16px;font-weight:bold;color:#000;padding:8px;")
        dh.setAlignment(Qt.AlignCenter)
        dtl.addWidget(dh)
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(8)
        self.data_table.setHorizontalHeaderLabels([
            "Label","Pathway","Electronic (Eh)","Enthalpy (Eh)",
            "Gibbs (Eh)","Rel. Energy","Files","Color"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setStyleSheet("""
            QTableWidget { border:1px solid #e5e7eb; border-radius:8px;
                background:white; gridline-color:#e5e7eb; font-size:12px; }
            QHeaderView::section { background:#f3f4f6; padding:8px;
                border:1px solid #e5e7eb; font-weight:bold; }
            QTableWidget::item { padding:6px; }
            QTableWidget::item:selected { background:#3b82f6; color:white; }
        """)
        tbl_btns = QHBoxLayout()
        for label, fn in [
            ("↑ Up",    lambda: self.move_table_item(-1)),
            ("↓ Down",  lambda: self.move_table_item(1)),
            ("Delete",  self.remove_table_items),
            ("CSV",     self.export_csv),
        ]:
            b = QPushButton(label); b.setMinimumHeight(32)
            b.clicked.connect(fn); tbl_btns.addWidget(b)
        tbl_btns.addStretch()
        dtl.addWidget(self.data_table)
        dtl.addLayout(tbl_btns)
        self.main_view_tabs.addTab(self.data_tab, "📋 Thermodynamic Table")

        # Canvas events
        self.canvas.mpl_connect('button_press_event',   self.on_press)
        self.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.splitter.addWidget(self.main_view_tabs)
        self.splitter.setStretchFactor(0, 1)

    # ─────────────────────────────────────────────────────────────────
    # TOOLBAR & MENU
    # ─────────────────────────────────────────────────────────────────
    def setup_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(26, 26))
        self.addToolBar(tb)
        # Floating controls-panel toggle — always reachable from the main window
        a_ctrl = QAction("\u2699 Controls", self)
        a_ctrl.setToolTip("Show / hide the floating Controls panel  (Ctrl+B)")
        a_ctrl.triggered.connect(self.toggle_sidebar)
        tb.addAction(a_ctrl)
        tb.addSeparator()
        for label, slot in [
            ("New",           self.new_session),
            ("Load Session",  self.load_session),
            ("Save Session",  self.save_session),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); tb.addAction(a)
        tb.addSeparator()
        for label, slot in [
            ("Update Plot",    self.update_plot),
            ("Copy 1200 DPI",  self.copy_to_clipboard),
            ("Export Graphic", self.export_image),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); tb.addAction(a)
        tb.addSeparator()
        for label, slot in [
            ("Align Labels",    self.auto_align_labels),
            ("Reset Labels",    self.reset_all_labels),
            ("Dark Theme",      lambda: self.apply_quick_style("Dark")),
            ("Light Theme",     lambda: self.apply_quick_style("Light")),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); tb.addAction(a)

    def setup_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet("QMenuBar { background-color:#f9fafb; }")

        fm = mb.addMenu("&File")
        for label, shortcut, slot in [
            ("&New Session",     QKeySequence.New,  self.new_session),
            ("&Load Session…",   QKeySequence.Open, self.load_session),
            ("&Save Session…",   QKeySequence.Save, self.save_session),
        ]:
            a = QAction(label, self); a.setShortcut(shortcut)
            a.triggered.connect(slot); fm.addAction(a)
        fm.addSeparator()
        for label, slot in [("&Import CSV…", self.import_data), ("&Export CSV…", self.export_csv)]:
            a = QAction(label, self); a.triggered.connect(slot); fm.addAction(a)
        fm.addSeparator()
        for label, shortcut, slot in [
            ("Export &Graphic…", "", self.export_image),
            ("&Copy 1200 DPI",   "Ctrl+Shift+C", self.copy_to_clipboard),
        ]:
            a = QAction(label, self)
            if shortcut: a.setShortcut(shortcut)
            a.triggered.connect(slot); fm.addAction(a)
        fm.addSeparator()
        ex = QAction("E&xit", self); ex.setShortcut(QKeySequence.Quit)
        ex.triggered.connect(self.close); fm.addAction(ex)

        em = mb.addMenu("&Edit")
        for label, shortcut, slot in [
            ("&Undo",        QKeySequence.Undo,      self.perform_undo),
            ("&Redo",        QKeySequence.Redo,      self.perform_redo),
        ]:
            a = QAction(label, self); a.setShortcut(shortcut)
            a.triggered.connect(slot); em.addAction(a)
        em.addSeparator()
        for label, slot in [
            ("Select &All",    self.select_all_states),
            ("&Align Labels",  self.auto_align_labels),
            ("&Reset Positions", self.reset_all_labels),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); em.addAction(a)

        vm = mb.addMenu("&View")
        for label, style in [("&Dark","Dark"),("&Light","Light"),("&Colorful","Colorful")]:
            a = QAction(label, self); a.triggered.connect(lambda c, s=style: self.apply_quick_style(s))
            vm.addAction(a)
        vm.addSeparator()
        ts = QAction("Toggle &Sidebar", self); ts.setShortcut("Ctrl+B")
        ts.triggered.connect(self.toggle_sidebar); vm.addAction(ts)

        hm = mb.addMenu("&Help")
        ab = QAction("&About", self); ab.triggered.connect(self.show_about); hm.addAction(ab)

    # ─────────────────────────────────────────────────────────────────
    # FLOATING CONTROL WINDOW HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _show_float_ctrl(self):
        """Position and show the floating control window beside/near the main window."""
        mg   = self.frameGeometry()
        ctrl_w = self.float_ctrl_win.width()
        ctrl_h = min(self.float_ctrl_win.height(), mg.height())
        screen  = QApplication.primaryScreen().availableGeometry()
        # Prefer right side; fall back to left if no room
        x = mg.right() + 8
        if x + ctrl_w > screen.right():
            x = max(screen.left(), mg.left() - ctrl_w - 8)
        y = max(screen.top(), mg.top())
        self.float_ctrl_win.move(x, y)
        self.float_ctrl_win.resize(ctrl_w, ctrl_h)
        self.float_ctrl_win.show()
        self.float_ctrl_win.raise_()

    def toggle_sidebar(self):
        """Show / hide the floating controls panel (Ctrl+B)."""
        if self.float_ctrl_win.isVisible():
            self.float_ctrl_win.hide()
        else:
            self.float_ctrl_win.show()
            self.float_ctrl_win.raise_()
            self.float_ctrl_win.activateWindow()

    # ─────────────────────────────────────────────────────────────────
    # PATHWAY HELPERS  ★ NEW
    # ─────────────────────────────────────────────────────────────────
    def _pathway_for(self, item: PlotItem) -> str:
        """Return the effective pathway name for an item."""
        return item.pathway.strip() if item.pathway.strip() else os.path.basename(item.folder_path)

    def _color_for(self, item: PlotItem) -> str:
        """Resolve display color: custom_color > pathway_color > folder_color."""
        if item.custom_color:
            return item.custom_color
        pw = self._pathway_for(item)
        if pw in self.pathway_colors:
            return self.pathway_colors[pw]
        return self.folders.get(item.folder_path, {'color': '#3b82f6'})['color']

    def _ensure_pathway_color(self, pw: str) -> str:
        """Guarantee a colour exists for *pw*, returning it."""
        if pw not in self.pathway_colors:
            self.pathway_colors[pw] = next(self.color_cycle)
        return self.pathway_colors[pw]

    def _all_pathways(self) -> List[str]:
        return sorted({self._pathway_for(it) for it in self.plot_staging_list})

    def refresh_pathway_list_widget(self):
        self.pw_list_widget.clear()
        for pw in self._all_pathways():
            color = self.pathway_colors.get(pw,
                self.folders.get(next((it.folder_path for it in self.plot_staging_list
                                       if self._pathway_for(it) == pw), ''),
                                  {'color': '#3b82f6'})['color'])
            item = QListWidgetItem(pw)
            item.setForeground(QColor(color))
            self.pw_list_widget.addItem(item)

    def set_pathway_for_selected(self):
        rows = [self.staging_list.row(i) for i in self.staging_list.selectedItems()]
        pw = self.pw_name_input.text().strip()
        if not rows:
            QMessageBox.warning(self, "No Selection", "Select states in the staging list first."); return
        if not pw:
            QMessageBox.warning(self, "No Name", "Enter a pathway name first."); return
        self.undo_mgr.push(self.plot_staging_list)
        self._ensure_pathway_color(pw)
        for r in rows:
            self.plot_staging_list[r].pathway = pw
        self._dirty = True
        self.sync_sidebar_list()
        self.update_plot()
        self.statusBar().showMessage(
            f"Assigned {len(rows)} state(s) to pathway '{pw}'", 3000)

    def choose_pathway_color(self):
        pw = self.pw_name_input.text().strip()
        if not pw:
            QMessageBox.warning(self, "No Name", "Enter a pathway name to colour."); return
        initial = QColor(self.pathway_colors.get(pw, "#3b82f6"))
        color = QColorDialog.getColor(initial, self)
        if color.isValid():
            self.pathway_colors[pw] = color.name()
            self._dirty = True
            self.refresh_pathway_list_widget()
            self.update_plot()

    # ─────────────────────────────────────────────────────────────────
    # CROSS-PATHWAY DIFFERENCES  ★ NEW
    # ─────────────────────────────────────────────────────────────────
    def refresh_xp_state_combos(self):
        """Repopulate the state-A / state-B combo boxes from current staging list."""
        for combo in (self.xp_state_a, self.xp_state_b):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for it in self.plot_staging_list:
                name = it.custom_name or it.display_name
                pw   = self._pathway_for(it)
                combo.addItem(f"[{pw}]  {name}")
            idx = combo.findText(prev)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def add_cross_diff(self):
        ia = self.xp_state_a.currentIndex()
        ib = self.xp_state_b.currentIndex()
        if ia < 0 or ib < 0 or ia == ib:
            QMessageBox.warning(self, "Invalid Selection",
                "Select two different states for comparison."); return
        label     = self.xp_label_input.text().strip() or ""
        arrow_sty = self.xp_arrow_combo.currentText()
        color     = '#e11d48'     # default hot-pink; could expose color picker
        entry = dict(ia=ia, ib=ib, label=label,
                     arrow=arrow_sty, color=color, offset=[0.0, 0.0])
        self.cross_diffs.append(entry)
        na = self.xp_state_a.currentText()
        nb = self.xp_state_b.currentText()
        li = QListWidgetItem(f"{na}  ↔  {nb}")
        li.setForeground(QColor(color))
        li.setData(Qt.UserRole, len(self.cross_diffs)-1)
        self.xp_list.addItem(li)
        self._dirty = True
        self.update_plot()
        self.statusBar().showMessage("Cross-pathway comparison added", 3000)

    def clear_cross_diffs(self):
        self.cross_diffs.clear()
        self.xp_list.clear()
        self._dirty = True
        self.update_plot()

    def _xp_context_menu(self, pos):
        item = self.xp_list.itemAt(pos)
        if item is None: return
        menu = QMenu()
        act_del  = menu.addAction("Remove this comparison")
        act_col  = menu.addAction("Change colour…")
        action   = menu.exec_(self.xp_list.viewport().mapToGlobal(pos))
        row = self.xp_list.row(item)
        if action == act_del:
            self.cross_diffs.pop(row)
            self.xp_list.takeItem(row)
            self._dirty = True
            self.update_plot()
        elif action == act_col:
            col = QColorDialog.getColor(
                QColor(self.cross_diffs[row].get('color','#e11d48')), self)
            if col.isValid():
                self.cross_diffs[row]['color'] = col.name()
                item.setForeground(col)
                self._dirty = True
                self.update_plot()

    # ─────────────────────────────────────────────────────────────────
    # STAGING INTERACTIONS
    # ─────────────────────────────────────────────────────────────────
    def show_staging_context_menu(self, position):
        menu = QMenu()
        menu.addAction("Edit Label & Pathway…").triggered.connect(self.open_label_editor)
        menu.addAction("Duplicate").triggered.connect(self.duplicate_selected_state)
        menu.addSeparator()
        menu.addAction("Move Up").triggered.connect(lambda: self.move_staging_item(-1))
        menu.addAction("Move Down").triggered.connect(lambda: self.move_staging_item(1))
        menu.addSeparator()
        menu.addAction("Remove").triggered.connect(self.remove_staged_items)
        menu.exec_(self.staging_list.viewport().mapToGlobal(position))

    def open_label_editor(self):
        row = self.staging_list.currentRow()
        if row == -1:
            QMessageBox.warning(self, "No Selection", "Select a state to edit."); return
        self.undo_mgr.push(self.plot_staging_list)
        all_pws = self._all_pathways()
        dlg = LabelEditorDialog(self.plot_staging_list[row], all_pws, self)
        if dlg.exec():
            values = dlg.get_values()
            for k, v in values.items():
                setattr(self.plot_staging_list[row], k, v)
            # Ensure pathway has a colour
            pw = self.plot_staging_list[row].pathway.strip()
            if pw:
                self._ensure_pathway_color(pw)
            self._dirty = True
            self.sync_sidebar_list()
            self.update_plot()

    def duplicate_selected_state(self):
        row = self.staging_list.currentRow()
        if row == -1: return
        self.undo_mgr.push(self.plot_staging_list)
        dup = copy.deepcopy(self.plot_staging_list[row])
        dup.custom_name = f"{dup.custom_name or dup.display_name} (copy)"
        dup.label_offset = [dup.label_offset[0]+0.1, dup.label_offset[1]+0.1]
        self.plot_staging_list.insert(row+1, dup)
        self._dirty = True
        self.sync_sidebar_list()
        self.update_plot()

    def filter_staging_list(self):
        q = self.staging_search.text().lower()
        for i in range(self.staging_list.count()):
            item = self.staging_list.item(i)
            item.setHidden(bool(q) and q not in item.text().lower())

    def on_staging_selection_changed(self):
        self.selected_states = {self.staging_list.row(i)
                                for i in self.staging_list.selectedItems()}

    def select_all_states(self):   self.staging_list.selectAll()
    def toggle_sidebar(self):
        self.control_panel.setVisible(not self.control_panel.isVisible())

    def sync_sidebar_list(self):
        self.staging_list.clear()
        for it in self.plot_staging_list:
            name    = it.custom_name or it.display_name
            pw      = self._pathway_for(it)
            display = f"[{pw}]  {name}"
            li = QListWidgetItem(display)
            c = self._color_for(it)
            li.setForeground(QColor(c))
            files_info = "\n".join([f"{m}× {os.path.basename(p)}" for p, m in it.file_data])
            li.setToolTip(f"Pathway: {pw}\nFiles:\n{files_info}")
            li.setData(Qt.UserRole, it)
            self.staging_list.addItem(li)
        self.filter_staging_list()
        self.refresh_pathway_list_widget()
        self.refresh_xp_state_combos()

    # ─────────────────────────────────────────────────────────────────
    # FOLDER LOADING
    # ─────────────────────────────────────────────────────────────────
    def add_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select ORCA Results Folder")
        if not path or path in self.folders: return
        progress = QProgressDialog("Scanning folder…", "Cancel", 0, 0, self)
        progress.setWindowTitle("Loading Folder")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        loader = FolderLoader(path)
        self._active_loaders.append(loader)          # keep alive
        loader.files_found.connect(lambda n: progress.setLabelText(
            f"Scanning folder… ({n} files found)"))
        loader.finished.connect(lambda d: self._on_folder_loaded(d, progress))
        loader.error.connect(lambda e: self._on_folder_error(e, progress))
        progress.canceled.connect(loader.stop)
        # Remove from active list (and schedule Qt-side cleanup) when done
        loader.finished.connect(lambda _d, l=loader: self._release_loader(l))
        loader.error.connect(lambda _e, l=loader:   self._release_loader(l))
        loader.start()

    def _release_loader(self, loader: 'FolderLoader'):
        """Remove a finished loader from the keep-alive list and schedule Qt cleanup."""
        try:
            self._active_loaders.remove(loader)
        except ValueError:
            pass
        loader.deleteLater()

    def _on_folder_loaded(self, data, progress): progress.close(); self.on_folder_loaded(data)
    def _on_folder_error(self, msg, progress):
        progress.close()
        QMessageBox.critical(self, "Error", f"Failed to load folder: {msg}")

    def on_folder_loaded(self, data):
        path = data['path']
        name = os.path.basename(path)
        color = next(self.color_cycle)

        tab = QWidget()
        tl  = QVBoxLayout(tab)

        sl = QHBoxLayout()
        se = QLineEdit(); se.setPlaceholderText("Search in this folder…")
        sl.addWidget(QLabel("🔍")); sl.addWidget(se)
        tl.addLayout(sl)

        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lst.setStyleSheet("QListWidget{border:1px solid #e5e7eb;border-radius:8px;"
                          "background:white;font-size:12px;}")

        file_items = []
        for d, f in data['files']:
            item = QListWidgetItem(d)
            item.setData(Qt.UserRole, f)
            item.setToolTip(f)
            lst.addItem(item)
            file_items.append((d, f, item))

        se.textChanged.connect(
            lambda t: [fi.setHidden(t.lower() not in fd.lower()) for fd, _ff, fi in file_items])

        bl = QHBoxLayout()
        b_add = GradientButton("Add Selected",   color1="#10b981", color2="#059669")
        b_add.clicked.connect(lambda: self.add_to_staging(path))
        b_sum = GradientButton("Add Summed",     color1="#06b6d4", color2="#0891b2")
        b_sum.clicked.connect(lambda: self.add_to_staging(path, combined=True))
        # ★ NEW: "Add as Pathway…" button lets users name a cascade at add time
        b_pw  = GradientButton("Add as Pathway…", color1="#8b5cf6", color2="#7c3aed")
        b_pw.setToolTip("Add selected files as a new named pathway — ideal for multiple cascades from the same folder.")
        b_pw.clicked.connect(lambda: self.add_to_staging_as_pathway(path))
        bl.addWidget(b_add); bl.addWidget(b_sum); bl.addWidget(b_pw)
        tl.addWidget(lst); tl.addLayout(bl)

        self.tab_folders.addTab(tab, name)
        self.folders[path] = {'color': color, 'list_widget': lst,
                              'files': data['files'], 'tab': tab}
        self.statusBar().showMessage(
            f"Loaded folder: {name} ({len(data['files'])} files)", 3000)

    # ─────────────────────────────────────────────────────────────────
    # ADDING TO STAGING
    # ─────────────────────────────────────────────────────────────────
    def add_to_staging(self, folder_path, combined=False):
        fd  = self.folders[folder_path]
        lst = fd['list_widget']
        sel = lst.selectedItems()
        if not sel:
            QMessageBox.information(self, "No Selection", "Please select files."); return

        default_pw = os.path.basename(folder_path)
        self._ensure_pathway_color(default_pw)

        if combined:
            paths = [i.data(Qt.UserRole) for i in sel]
            dlg   = MultipliersDialog(paths, self)
            if dlg.exec():
                fd_vals = dlg.get_values()
                disp = " + ".join([f"{m:.1f}×{os.path.basename(p)[:8]}" for p, m in fd_vals])
                it = PlotItem(file_data=fd_vals, display_name=disp,
                              folder_path=folder_path, pathway=default_pw)
                self.plot_staging_list.append(it)
                li = QListWidgetItem(f"[{default_pw}]  {disp}")
                li.setForeground(QColor(self.pathway_colors[default_pw]))
                self.staging_list.addItem(li)
        else:
            for i in sel:
                full = i.data(Qt.UserRole)
                disp = f"[{os.path.basename(folder_path)}] {i.text()}"
                it   = PlotItem(file_data=[(full, 1.0)], display_name=disp,
                                folder_path=folder_path, pathway=default_pw)
                self.plot_staging_list.append(it)
                li = QListWidgetItem(f"[{default_pw}]  {disp}")
                li.setForeground(QColor(self.pathway_colors[default_pw]))
                self.staging_list.addItem(li)

        self.filter_staging_list()
        self.refresh_pathway_list_widget()
        self.update_plot()
        self.statusBar().showMessage(f"Added {len(sel)} item(s) to staging", 3000)

    def add_to_staging_as_pathway(self, folder_path):
        """★ Add selected files under a user-named pathway (multiple cascades from one folder)."""
        fd  = self.folders[folder_path]
        lst = fd['list_widget']
        sel = lst.selectedItems()
        if not sel:
            QMessageBox.information(self, "No Selection", "Please select files."); return

        existing_pws = self._all_pathways()
        default_name = f"{os.path.basename(folder_path)} ({len(existing_pws)+1})"
        pw, ok = QInputDialog.getText(
            self, "New Pathway Name",
            "Enter a name for this cascade / pathway:\n"
            f"(existing: {', '.join(existing_pws) or 'none'})",
            text=default_name)
        if not ok or not pw.strip(): return
        pw = pw.strip()
        self._ensure_pathway_color(pw)

        for i in sel:
            full = i.data(Qt.UserRole)
            disp = f"[{os.path.basename(folder_path)}] {i.text()}"
            it   = PlotItem(file_data=[(full, 1.0)], display_name=disp,
                            folder_path=folder_path, pathway=pw)
            self.plot_staging_list.append(it)
            li = QListWidgetItem(f"[{pw}]  {disp}")
            li.setForeground(QColor(self.pathway_colors[pw]))
            self.staging_list.addItem(li)

        self.filter_staging_list()
        self.refresh_pathway_list_widget()
        self.update_plot()
        self.statusBar().showMessage(
            f"Added {len(sel)} item(s) to pathway '{pw}'", 3000)

    def open_global_browser(self):
        if not self.folders:
            QMessageBox.information(self, "Empty", "Add a folder first."); return
        dlg = GlobalFileBrowser(self.folders, self)
        if dlg.exec():
            for folder_path, full_path in dlg.selected_files:
                pw   = self._pathway_for(PlotItem([], "", folder_path))
                pw   = os.path.basename(folder_path)
                self._ensure_pathway_color(pw)
                disp = f"[{os.path.basename(folder_path)}] {os.path.basename(full_path)}"
                it   = PlotItem(file_data=[(full_path, 1.0)], display_name=disp,
                                folder_path=folder_path, pathway=pw)
                self.plot_staging_list.append(it)
                li   = QListWidgetItem(f"[{pw}]  {disp}")
                li.setForeground(QColor(self.pathway_colors[pw]))
                self.staging_list.addItem(li)
            self.filter_staging_list()
            self.refresh_pathway_list_widget()
            self.update_plot()
            self.statusBar().showMessage(
                f"Added {len(dlg.selected_files)} files", 3000)

    # ─────────────────────────────────────────────────────────────────
    # ENERGY CALCULATION
    # ─────────────────────────────────────────────────────────────────
    def _conv_factor(self):
        if self.btn_kcal.isChecked(): return 627.509
        if self.btn_kj.isChecked():   return 2625.5
        return 27.2114

    def calculate_state_energy(self, item: PlotItem, energy_type: str) -> Optional[float]:
        if item.energy is not None and not item.file_data:
            return item.energy
        total = 0.0
        for path, mult in item.file_data:
            val = get_cached_parser(path).get_energy(energy_type)
            if val is None: return None
            total += val * mult
        return total

    def get_reference_energy(self) -> Optional[float]:
        e_type  = self.energy_type_combo.currentText().lower()
        ref_txt = self.zero_ref_combo.currentText()
        if ref_txt == "Per Pathway (First State)":
            return None
        if ref_txt == "Minimum (Global)":
            vals = [self.calculate_state_energy(it, e_type)
                    for it in self.plot_staging_list]
            valid = [v for v in vals if v is not None]
            return np.min(valid) if valid else None
        # Named state
        for it in self.plot_staging_list:
            if (it.custom_name or it.display_name) == ref_txt:
                return self.calculate_state_energy(it, e_type)
        return None

    def update_zero_reference_combo(self):
        self.zero_ref_combo.blockSignals(True)
        current = self.zero_ref_combo.currentText()
        self.zero_ref_combo.clear()
        self.zero_ref_combo.addItem("Minimum (Global)")
        self.zero_ref_combo.addItem("Per Pathway (First State)")
        for it in self.plot_staging_list:
            self.zero_ref_combo.addItem(it.custom_name or it.display_name)
        idx = self.zero_ref_combo.findText(current)
        if idx >= 0: self.zero_ref_combo.setCurrentIndex(idx)
        self.zero_ref_combo.blockSignals(False)

    def update_energy_units(self):
        sender = self.sender()
        for b in [self.btn_kcal, self.btn_kj, self.btn_ev]: b.setChecked(b is sender)
        if sender == self.btn_kcal: self.y_label_input.setText("Relative Energy (kcal/mol)")
        elif sender == self.btn_kj: self.y_label_input.setText("Relative Energy (kJ/mol)")
        else:                       self.y_label_input.setText("Relative Energy (eV)")
        self.update_plot()

    # ─────────────────────────────────────────────────────────────────
    # INTERACTIVE DRAG
    # ─────────────────────────────────────────────────────────────────
    def _get_display_y(self, item: PlotItem) -> Optional[float]:
        e_type = self.energy_type_combo.currentText().lower()
        conv   = self._conv_factor()
        ref_mode = self.zero_ref_combo.currentText()
        val = self.calculate_state_energy(item, e_type)
        if val is None: return None
        if ref_mode == "Per Pathway (First State)":
            pw = self._pathway_for(item)
            first_val = next(
                (self.calculate_state_energy(it, e_type)
                 for it in self.plot_staging_list if self._pathway_for(it) == pw),
                None)
            if first_val is None: return None
            return (val - first_val) * conv
        else:
            ref = self.get_reference_energy()
            return (val - ref) * conv if ref is not None else None

    def _build_x_map(self):
        """Return {orig_idx: lx} using the same layout logic as update_plot."""
        is_ov  = self.check_overlay.isChecked()
        gap    = self.pathway_gap_spin.value()
        result = {}
        # Group consecutive items by pathway
        groups = []
        for idx, it in enumerate(self.plot_staging_list):
            pw = self._pathway_for(it)
            if groups and groups[-1][0] == pw:
                groups[-1][1].append((idx, it))
            else:
                groups.append((pw, [(idx, it)]))

        curr_x = 0
        for pw, items in groups:
            for i, (orig_idx, _) in enumerate(items):
                lx = i if is_ov else (curr_x + i)
                result[orig_idx] = lx
            if not is_ov:
                curr_x += len(items) + gap
        return result

    def _data_to_px(self, x_data: float, y_data: float) -> Tuple[float, float]:
        """Convert a single data-coordinate point to display (pixel) coords."""
        arr = self.ax.transData.transform([[x_data, y_data]])
        return float(arr[0, 0]), float(arr[0, 1])

    def on_press(self, event):
        if not self.check_interactive.isChecked() or event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return

        # Hit-test entirely in PIXEL space so x/y scale mismatch is irrelevant.
        PX_THRESH_LBL = 28   # pixels — generous enough to click on text
        PX_THRESH_BAR = 20   # pixels — for clicking the bar itself

        px_click, py_click = self._data_to_px(event.xdata, event.ydata)

        min_dist_px  = 1e9
        found_idx    = None
        found_type   = None
        found_delta_key = None

        # 1 ── Energy-label positions (stored in data coords → convert to px)
        for idx, (lx_r, ly_r) in self._label_plot_pos.items():
            px_lbl, py_lbl = self._data_to_px(lx_r, ly_r)
            d = np.hypot(px_click - px_lbl, py_click - py_lbl)
            if d < PX_THRESH_LBL and d < min_dist_px:
                min_dist_px = d; found_idx = idx; found_type = 'label'

        # 2 ── Delta-label positions
        for (i1, i2), (dx_r, dy_r) in self._delta_plot_pos.items():
            px_d, py_d = self._data_to_px(dx_r, dy_r)
            d = np.hypot(px_click - px_d, py_click - py_d)
            if d < PX_THRESH_LBL and d < min_dist_px:
                min_dist_px = d; found_idx = i1; found_type = 'delta'
                found_delta_key = (i1, i2)

        # 3 ── State bars (fallback — click on the bar line itself)
        x_map = self._build_x_map()
        hw = self.state_width_slider.value() / 2.0
        for orig_idx, it in enumerate(self.plot_staging_list):
            lx = x_map.get(orig_idx)
            if lx is None: continue
            y = self._get_display_y(it)
            if y is None: continue
            # Convert bar extents to px
            px_l, py_bar = self._data_to_px(lx - hw, y)
            px_r, _      = self._data_to_px(lx + hw, y)
            # horizontal: click inside bar width; vertical: within PX_THRESH_BAR px
            if px_l <= px_click <= px_r and abs(py_click - py_bar) < PX_THRESH_BAR:
                d = abs(py_click - py_bar)
                if d < min_dist_px and found_type not in ('label', 'delta'):
                    min_dist_px = d
                    found_idx = orig_idx; found_type = 'state'
                    found_delta_key = None

        self.dragging_idx        = found_idx
        self.dragging_type       = found_type
        self._dragging_delta_key = found_delta_key
        if found_idx is not None:
            self.undo_mgr.push(self.plot_staging_list)
            kind = {'label': 'energy label', 'delta': 'Δ label',
                    'state': 'state bar'}.get(found_type, '')
            name = (self.plot_staging_list[found_idx].custom_name
                    or self.plot_staging_list[found_idx].display_name)
            self.statusBar().showMessage(f"Dragging {kind}: {name}", 2000)

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return

        it    = self.plot_staging_list[self.dragging_idx]
        x_map = self._build_x_map()
        lx    = x_map.get(self.dragging_idx)
        if lx is None: return
        y = self._get_display_y(it)
        if y is None: return

        if self.dragging_type == 'label':
            # Store the raw offset — smart_y_offset will be bypassed for this item
            it.label_offset[0] = event.xdata - lx
            it.label_offset[1] = event.ydata - y

        elif self.dragging_type == 'delta' and self._dragging_delta_key is not None:
            i1, i2 = self._dragging_delta_key
            lx1 = x_map.get(i1, lx)
            lx2 = x_map.get(i2, lx + 1)
            y1  = self._get_display_y(self.plot_staging_list[i1])
            y2  = self._get_display_y(self.plot_staging_list[i2])
            if y1 is not None and y2 is not None:
                mid_x = (lx1 + lx2) / 2.0
                mid_y = (y1  + y2)  / 2.0
                self.plot_staging_list[i1].delta_offset[0] = event.xdata - mid_x
                self.plot_staging_list[i1].delta_offset[1] = event.ydata - mid_y

        elif self.dragging_type == 'state':
            it.label_offset[0] = event.xdata - lx

        self.update_plot()
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging_idx is not None:
            self._dirty = True
            self.statusBar().showMessage("Position saved  ·  Ctrl+Z to undo", 3000)
        self.dragging_idx        = None
        self.dragging_type       = None
        self._dragging_delta_key = None

    # ─────────────────────────────────────────────────────────────────
    # ★ IMPROVED PLOTTING ENGINE
    # ─────────────────────────────────────────────────────────────────
    def update_plot(self):
        if not self.plot_staging_list:
            self.ax.clear(); self.canvas.draw(); return

        self.update_data_table()
        self.update_zero_reference_combo()

        # Theme
        theme = self.theme_combo.currentText()
        mpl_style.use('default' if theme == "Default" else theme)
        is_pub = self.check_pub_quality.isChecked()
        matplotlib.rcParams.update({
            'font.family':      'serif'      if is_pub else 'sans-serif',
            'font.serif':       ['Times New Roman','DejaVu Serif','serif'],
            'font.sans-serif':  ['Arial','DejaVu Sans','sans-serif'],
            'font.size':        self.base_fs_spin.value(),
            'xtick.direction':  'in' if self.check_inward_ticks.isChecked() else 'out',
            'ytick.direction':  'in' if self.check_inward_ticks.isChecked() else 'out',
            'axes.linewidth':   1.2,
            'axes.edgecolor':   '#495057',
            'axes.facecolor':   'white',
        })

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.fig.set_size_inches(self.fig_width_spin.value(), self.fig_height_spin.value())
        self.fig.subplots_adjust(left=self.margin_left_spin.value(),
                                 right=0.95, top=0.90, bottom=0.18)
        self.fig.set_facecolor('#f9fafb' if not self.check_shadow.isChecked() else 'white')

        e_type   = self.energy_type_combo.currentText().lower()
        conv     = self._conv_factor()
        ref_mode = self.zero_ref_combo.currentText()
        hw       = self.state_width_slider.value() / 2.0
        gap      = self.pathway_gap_spin.value()
        ls_str   = _LINESTYLE_MAP.get(self.line_style_combo.currentText(), '-')
        bar_thick= self.bar_thickness_spin.value()
        t_fs     = self.tick_fs_spin.value()
        t_b      = 'bold' if self.check_tick_bold.isChecked() else 'normal'
        e_fs     = self.energy_fs_spin.value()
        e_b      = 'bold' if self.check_energy_bold.isChecked() else 'normal'
        d_fs     = self.delta_fs_spin.value()
        d_b      = 'bold' if self.check_delta_bold.isChecked() else 'normal'
        rot      = self.xlabel_rotation_spin.value()
        label_mode = self.label_mode_combo.currentText()
        is_ov    = self.check_overlay.isChecked()
        smart    = self.check_smart_labels.isChecked()
        global_ref = None if ref_mode == "Per Pathway (First State)" else self.get_reference_energy()

        # ── Build pathway groups (consecutive items sharing same pathway) ──
        groups: List[Tuple[str, List[Tuple[int, PlotItem]]]] = []
        for idx, it in enumerate(self.plot_staging_list):
            pw = self._pathway_for(it)
            if groups and groups[-1][0] == pw:
                groups[-1][1].append((idx, it))
            else:
                groups.append((pw, [(idx, it)]))

        # ── Assign X positions ──
        curr_x = 0.0
        x_map: Dict[int, float] = {}      # orig_idx → lx
        pw_first_x: Dict[str, float] = {} # pathway → start x
        for pw, items in groups:
            for i, (orig_idx, _) in enumerate(items):
                lx = i if is_ov else (curr_x + i)
                x_map[orig_idx] = lx
            if pw not in pw_first_x:
                pw_first_x[pw] = x_map[items[0][0]]
            if not is_ov:
                curr_x += len(items) + gap

        max_x = max(x_map.values()) if x_map else 0
        tick_pos: List[float] = []
        tick_labels: List[str] = []

        # Pre-compute all y values for smart label placement
        all_ys: Dict[int, float] = {}
        for idx, it in enumerate(self.plot_staging_list):
            pw = self._pathway_for(it)
            if ref_mode == "Per Pathway (First State)":
                first_items = next(
                    (items for g_pw, items in groups if g_pw == pw), [])
                first_val = (self.calculate_state_energy(first_items[0][1], e_type)
                             if first_items else None)
                folder_ref = first_val
            else:
                folder_ref = global_ref
            val = self.calculate_state_energy(it, e_type)
            if val is not None and folder_ref is not None:
                all_ys[idx] = (val - folder_ref) * conv

        # Smart label registry: maps rounded-x → list of used y positions
        label_registry: Dict[float, List[float]] = {}

        def smart_y_offset(lx, y, default_dy, y_range, item_idx=None):
            """Return a dy that avoids crowding.
            Bypasses avoidance for the item currently being dragged so the
            label follows the mouse exactly without snapping away."""
            # Always respect the user's manual offset for the dragged item
            if item_idx is not None and item_idx == self.dragging_idx:
                return default_dy
            if not smart or y_range == 0:
                return default_dy
            sep  = 0.13 * y_range
            cand = y + default_dy
            key  = round(lx, 1)
            used = label_registry.get(key, [])
            for _ in range(12):
                if all(abs(cand - u) >= sep for u in used):
                    break
                cand += sep
            label_registry.setdefault(key, []).append(cand)
            return cand - y

        # ── Draw ──
        # Reset stored label/delta positions for this render
        self._label_plot_pos.clear()
        self._delta_plot_pos.clear()

        global_conn_style   = self.conn_style_combo.currentText()
        global_extra_thermo = self.extra_thermo_display_combo.currentText()
        lbl_box_style       = self.label_box_combo.currentText()   # "Rounded Box"/"Square Box"/"No Box"
        dlt_box_style       = self.delta_box_combo.currentText()
        dlt_arrow_style     = self.delta_arrow_combo.currentText() # "Straight →"/"Wavy"/etc/"No Arrow"
        y_range_est         = (max(all_ys.values()) - min(all_ys.values())
                               if len(all_ys) > 1 else 1.0)

        def _make_bbox(style, face='white', edge_c='#333', alpha=0.85, lw=0.7):
            """Build a bbox dict or None based on combo choice."""
            if style == "No Box":
                return None
            bs = "round,pad=0.25" if style == "Rounded Box" else "square,pad=0.25"
            return dict(boxstyle=bs, facecolor=face, edgecolor=edge_c,
                        alpha=alpha, linewidth=lw)

        for pw, items in groups:
            # Determine reference for this pathway
            if ref_mode == "Per Pathway (First State)":
                first_val  = self.calculate_state_energy(items[0][1], e_type)
                folder_ref = first_val
            else:
                folder_ref = global_ref

            pw_color = self._ensure_pathway_color(pw)
            count    = len(items)

            for i, (orig_idx, it) in enumerate(items):
                lx  = x_map[orig_idx]
                val = self.calculate_state_energy(it, e_type)
                if val is None or folder_ref is None: continue
                y   = (val - folder_ref) * conv
                draw_c = self._color_for(it)

                # ── Bar ──
                if self.check_shadow.isChecked():
                    self.ax.plot([lx-hw, lx+hw], [y, y],
                                 color='gray', lw=bar_thick+1.5, alpha=0.25,
                                 zorder=4, solid_capstyle='butt')
                self.ax.plot([lx-hw, lx+hw], [y, y],
                             color=draw_c, lw=bar_thick, zorder=5, solid_capstyle='butt')

                # ── Energy value label ──
                if self.check_show_e_labels.isChecked():
                    dx_off, dy_raw = it.label_offset
                    dy      = smart_y_offset(lx + dx_off, y, dy_raw, y_range_est,
                                               item_idx=orig_idx)
                    lbl_c   = it.label_font_color or draw_c
                    lbl_fs  = it.label_font_size  or e_fs
                    lbl_fw  = it.label_font_weight if it.label_font_weight != 'normal' else e_b
                    lbl_x   = lx + dx_off
                    lbl_y   = y  + dy
                    self._label_plot_pos[orig_idx] = (lbl_x, lbl_y)

                    e_bbox  = _make_bbox(lbl_box_style, edge_c=lbl_c)
                    txt_kw  = dict(ha='center', va='bottom',
                                   color=lbl_c, fontweight=lbl_fw, fontsize=lbl_fs,
                                   zorder=9)
                    if e_bbox:
                        txt_kw['bbox'] = e_bbox

                    # Show arrow pointing to bar when label has been displaced
                    is_displaced = (abs(dx_off) > 0.08 or
                                    abs(it.label_offset[1] - 0.5) > 0.25)
                    if is_displaced:
                        ap_kw = dict(arrowprops=dict(
                            arrowstyle='->', color=lbl_c,
                            lw=0.9, alpha=0.70, shrinkA=2, shrinkB=3))
                        self.ax.annotate(
                            f"{y:.1f}",
                            xy=(lx, y), xytext=(lbl_x, lbl_y),
                            **txt_kw, **ap_kw)
                    else:
                        self.ax.text(lbl_x, lbl_y, f"{y:.1f}", **txt_kw)

                    # ── Extra thermo display ──
                    show_et = it.show_extra_thermo or (global_extra_thermo != "None")
                    et_type = it.extra_thermo_type if it.show_extra_thermo else global_extra_thermo
                    if show_et and it.file_data:
                        et_lines = []
                        try:
                            thermo = get_cached_parser(it.file_data[0][0]).get_all_thermo()
                            if et_type in ("Enthalpy", "Both") and thermo.get('Enthalpy') is not None:
                                h_sum = sum(get_cached_parser(p).get_energy('enthalpy') * m
                                            for p, m in it.file_data
                                            if get_cached_parser(p).get_energy('enthalpy') is not None)
                                if ref_mode == "Per Pathway (First State)":
                                    h_ref = sum(get_cached_parser(p).get_energy('enthalpy') * m
                                                for p, m in items[0][1].file_data
                                                if get_cached_parser(p).get_energy('enthalpy') is not None)
                                else:
                                    h_vals = []
                                    for it2 in self.plot_staging_list:
                                        hv = sum(get_cached_parser(p).get_energy('enthalpy') * m
                                                 for p, m in it2.file_data
                                                 if get_cached_parser(p).get_energy('enthalpy') is not None)
                                        h_vals.append(hv)
                                    h_ref = min(h_vals) if h_vals else 0.0
                                et_lines.append(f"H={((h_sum-h_ref)*conv):+.1f}")
                            if et_type in ("Electronic", "Both") and thermo.get('Electronic') is not None:
                                e_sum = sum(get_cached_parser(p).get_energy('electronic') * m
                                            for p, m in it.file_data
                                            if get_cached_parser(p).get_energy('electronic') is not None)
                                et_lines.append(f"E={e_sum*conv:.1f}")
                        except Exception:
                            pass
                        if et_lines:
                            et_bbox = _make_bbox("Rounded Box", face='#fffde7',
                                                 edge_c=lbl_c, alpha=0.82, lw=0.5)
                            et_kw = dict(ha='center', va='top', color=lbl_c,
                                         fontsize=max(lbl_fs-2, 5), fontstyle='italic',
                                         zorder=9)
                            if et_bbox: et_kw['bbox'] = et_bbox
                            self.ax.text(
                                lbl_x,
                                lbl_y - 0.02 * (y_range_est or 1),
                                "\n".join(et_lines), **et_kw)

                # ── Inline state name labels ──
                state_name = it.custom_name or it.display_name
                if label_mode in ("Inline Below Bar", "Both (Tick + Inline)"):
                    offset_y = max(0.07 * y_range_est, 0.3)
                    ha_rot   = 'right' if rot > 20 else 'center'
                    sn_bbox  = _make_bbox("Rounded Box", edge_c=draw_c, alpha=0.75, lw=0.5)
                    sn_kw = dict(ha=ha_rot, va='top', fontsize=max(t_fs-1, 6),
                                 fontweight=t_b, color=draw_c, rotation=rot, zorder=8)
                    if sn_bbox: sn_kw['bbox'] = sn_bbox
                    self.ax.text(lx, y - offset_y, state_name, **sn_kw)
                elif label_mode == "Inline Above Bar":
                    offset_y = max(0.07 * y_range_est, 0.3)
                    ha_rot   = 'left' if rot > 20 else 'center'
                    sn_bbox  = _make_bbox("Rounded Box", edge_c=draw_c, alpha=0.75, lw=0.5)
                    sn_kw = dict(ha=ha_rot, va='bottom', fontsize=max(t_fs-1, 6),
                                 fontweight=t_b, color=draw_c, rotation=rot, zorder=8)
                    if sn_bbox: sn_kw['bbox'] = sn_bbox
                    self.ax.text(lx, y + offset_y, state_name, **sn_kw)

                # Tick positions
                if label_mode != "No State Labels":
                    if lx not in tick_pos:
                        tick_pos.append(lx)
                        tick_labels.append(
                            "" if label_mode in ("Inline Below Bar", "Inline Above Bar")
                            else state_name)

                # ── Connecting line + delta ──
                if i < count - 1:
                    next_orig_idx, next_it = items[i+1]
                    next_val = self.calculate_state_energy(next_it, e_type)
                    if next_val is not None and folder_ref is not None:
                        ny      = (next_val - folder_ref) * conv
                        next_lx = x_map[next_orig_idx]
                        x1, x2  = lx + hw, next_lx - hw
                        lw_conn = bar_thick * 0.45

                        eff_style = (it.conn_arrow_style
                                     if it.conn_arrow_style not in ("Default", "")
                                     else global_conn_style)
                        kw_line = dict(color=pw_color, ls=ls_str, alpha=0.60,
                                       lw=lw_conn, zorder=3)
                        if eff_style == "Straight":
                            self.ax.plot([x1, x2], [y, ny], **kw_line)
                        elif eff_style in ("Smooth (Bezier)", "Smooth"):
                            xs = np.linspace(x1, x2, 60)
                            ts = (xs - x1) / max(x2 - x1, 1e-9)
                            self.ax.plot(xs, y+(ny-y)*(3*ts**2-2*ts**3), **kw_line)
                        elif eff_style == "Wavy":
                            _draw_wavy(self.ax, x1, y, x2, ny, n_waves=6, amp_frac=0.04, **kw_line)
                        elif eff_style == "Zigzag":
                            _draw_zigzag(self.ax, x1, y, x2, ny, n_zigs=8, amp_frac=0.04, **kw_line)
                        elif eff_style == "Curled (Arc)":
                            _draw_curled(self.ax, x1, y, x2, ny, rad=0.30, **kw_line)
                        else:
                            xs = np.linspace(x1, x2, 60)
                            ts = (xs - x1) / max(x2 - x1, 1e-9)
                            self.ax.plot(xs, y+(ny-y)*(3*ts**2-2*ts**3), **kw_line)

                        # ── Delta label (draggable, styled arrow) ──
                        if self.check_show_delta.isChecked():
                            mid_x = (lx + next_lx) / 2.0
                            mid_y = (y  + ny) / 2.0
                            ddx, ddy = it.delta_offset
                            ann_x = mid_x + ddx
                            ann_y = mid_y + ddy
                            self._delta_plot_pos[(orig_idx, next_orig_idx)] = (ann_x, ann_y)
                            delta = ny - y
                            sign  = '+' if delta >= 0 else ''
                            d_bbox = _make_bbox(dlt_box_style, edge_c=pw_color,
                                                alpha=0.88, lw=0.7)
                            d_txt_kw = dict(ha='center', va='center', color=pw_color,
                                            fontsize=d_fs, fontweight=d_b, zorder=10)
                            if d_bbox: d_txt_kw['bbox'] = d_bbox

                            # Draw the text label
                            self.ax.text(ann_x, ann_y, fr"$\Delta={sign}{delta:.1f}$",
                                         **d_txt_kw)
                            # Draw styled arrow from label toward midpoint
                            if dlt_arrow_style != "No Arrow":
                                arr_style = dlt_arrow_style.replace(" →", "")
                                # Only draw arrow when displaced from midpoint
                                dist = np.hypot(ddx, ddy)
                                if dist > 0.05:
                                    _draw_fancy_arrow(
                                        self.ax,
                                        ann_x, ann_y, mid_x, mid_y,
                                        style=arr_style,
                                        color=pw_color, lw=0.9, alpha=0.75,
                                        ls='-', zorder=11)

            # Legend entry per pathway
            self.ax.plot([], [], color=pw_color, lw=2.5, label=pw)

        # ── Cross-pathway differences ★ NEW ──
        for diff in self.cross_diffs:
            ia = diff.get('ia', -1)
            ib = diff.get('ib', -1)
            if (ia < 0 or ib < 0 or
                    ia >= len(self.plot_staging_list) or
                    ib >= len(self.plot_staging_list)):
                continue
            ya = all_ys.get(ia)
            yb = all_ys.get(ib)
            lxa = x_map.get(ia)
            lxb = x_map.get(ib)
            if ya is None or yb is None or lxa is None or lxb is None:
                continue
            xp_c    = diff.get('color', '#e11d48')
            xp_lbl  = diff.get('label', '') or f"{yb-ya:+.1f}"
            xp_arr  = diff.get('arrow', 'Straight')
            xp_off  = diff.get('offset', [0.0, 0.0])

            # Vertical extent
            y_top   = max(ya, yb)
            y_bot   = min(ya, yb)
            x_mid   = (lxa + lxb) / 2.0 + xp_off[0]
            y_label = (ya + yb) / 2.0    + xp_off[1]

            if xp_arr == "Bracket":
                # Draw vertical bracket lines + horizontal caps
                x_bracket = x_mid
                cap_w = 0.08
                self.ax.plot([x_bracket, x_bracket], [ya, yb],
                             color=xp_c, lw=1.4, alpha=0.80, zorder=6,
                             ls='--')
                self.ax.plot([x_bracket-cap_w, x_bracket+cap_w], [ya, ya],
                             color=xp_c, lw=1.4, alpha=0.80, zorder=6)
                self.ax.plot([x_bracket-cap_w, x_bracket+cap_w], [yb, yb],
                             color=xp_c, lw=1.4, alpha=0.80, zorder=6)
            else:
                # Draw styled double-headed arrow along the side
                x_side = max(lxa, lxb) + hw + 0.25 + xp_off[0]
                _draw_fancy_arrow(self.ax, x_side, ya, x_side, yb,
                                  style=xp_arr, color=xp_c, lw=1.2,
                                  alpha=0.85, zorder=6)
                _draw_fancy_arrow(self.ax, x_side, yb, x_side, ya,
                                  style=xp_arr, color=xp_c, lw=1.2,
                                  alpha=0.85, zorder=6)
                x_label = x_side + 0.12

            # Label with highlighted box
            xp_text = f"{xp_lbl}\n({yb-ya:+.1f})" if xp_lbl and xp_lbl != f"{yb-ya:+.1f}" else f"{yb-ya:+.1f}"
            xp_bbox = dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor=xp_c, alpha=0.92, linewidth=1.0)
            self.ax.text(
                x_label if xp_arr != "Bracket" else x_mid + 0.10,
                y_label,
                xp_text,
                ha='left', va='center',
                color=xp_c, fontsize=max(d_fs-1, 6), fontweight='bold',
                zorder=12, bbox=xp_bbox)

        # ── Axes formatting ──
        self.ax.set_title(
            self.title_input.text() if self.title_input.text() != "None" else "",
            fontsize=self.base_fs_spin.value()+2, fontweight='bold', pad=15)
        self.ax.set_ylabel(self.y_label_input.text(),
                           fontsize=self.base_fs_spin.value(), labelpad=10)

        if label_mode == "No State Labels":
            self.ax.set_xticks([])
        else:
            self.ax.set_xticks(tick_pos)
            self.ax.set_xticklabels(tick_labels)
            for lbl in self.ax.get_xticklabels():
                lbl.set_fontsize(t_fs)
                lbl.set_fontweight(t_b)
                lbl.set_rotation(rot)
                if rot > 0:
                    lbl.set_ha('right')

        for lbl in self.ax.get_yticklabels():
            lbl.set_fontsize(t_fs)

        # Adjust bottom margin for rotated labels
        bottom_margin = 0.18 + (rot / 900)
        self.fig.subplots_adjust(left=self.margin_left_spin.value(),
                                 right=0.95, top=0.90, bottom=min(bottom_margin, 0.38))

        self.ax.set_xlim(-0.80, max_x + 0.80)
        y_min, y_max = self.ax.get_ylim()
        y_rng = y_max - y_min if y_max != y_min else 1.0
        self.ax.set_ylim(y_min - 0.10*y_rng, y_max + 0.15*y_rng)

        if self.check_grid.isChecked():
            self.ax.grid(True, axis='y', alpha=0.15, ls=':', color='gray', zorder=0)
            self.ax.grid(True, axis='x', alpha=0.05, ls=':', color='gray', zorder=0)

        if self.check_show_legend.isChecked() and any(it for it in self.plot_staging_list):
            leg = self.ax.legend(loc=self.legend_loc_combo.currentText(),
                                 frameon=True, fancybox=True,
                                 framealpha=0.92, edgecolor='#9ca3af',
                                 fontsize=max(t_fs-1, 6))
            leg.get_frame().set_linewidth(0.6)

        self.ax.get_xaxis().set_visible(self.check_show_xaxis.isChecked())
        self.ax.get_yaxis().set_visible(self.check_show_yaxis.isChecked())
        self.ax.set_frame_on(self.check_show_frame.isChecked())
        if self.check_shadow.isChecked():
            self.ax.set_facecolor('#f9fafb')

        self.canvas.draw()
        self.statusBar().showMessage("Plot updated", 2000)

    # ─────────────────────────────────────────────────────────────────
    # DATA TABLE
    # ─────────────────────────────────────────────────────────────────
    def update_data_table(self):
        self.data_table.setRowCount(len(self.plot_staging_list))
        e_type   = self.energy_type_combo.currentText().lower()
        conv     = self._conv_factor()
        ref_mode = self.zero_ref_combo.currentText()

        # Build per-pathway refs
        pw_refs: Dict[str, Optional[float]] = {}
        for it in self.plot_staging_list:
            pw = self._pathway_for(it)
            if pw not in pw_refs:
                pw_refs[pw] = self.calculate_state_energy(it, e_type)

        global_ref = self.get_reference_energy()

        for row, it in enumerate(self.plot_staging_list):
            name = it.custom_name or it.display_name
            pw   = self._pathway_for(it)

            thermo = {'Electronic': 0.0, 'Enthalpy': 0.0, 'Gibbs': 0.0}
            for path, mult in it.file_data:
                ft = get_cached_parser(path).get_all_thermo()
                for k in thermo:
                    if thermo[k] is None: continue
                    if ft.get(k) is None: thermo[k] = None
                    else:                 thermo[k] += ft[k] * mult

            self.data_table.setItem(row, 0, QTableWidgetItem(name))
            self.data_table.setItem(row, 1, QTableWidgetItem(pw))
            for col, key in enumerate(['Electronic','Enthalpy','Gibbs'], start=2):
                v = thermo[key]
                self.data_table.setItem(row, col,
                    QTableWidgetItem(f"{v:.6f}" if v is not None else "N/A"))

            val = self.calculate_state_energy(it, e_type)
            if ref_mode == "Per Pathway (First State)":
                ref = pw_refs.get(pw)
            else:
                ref = global_ref
            if ref is not None and val is not None:
                self.data_table.setItem(row, 5,
                    QTableWidgetItem(f"{(val - ref)*conv:.2f}"))
            else:
                self.data_table.setItem(row, 5, QTableWidgetItem("N/A"))

            self.data_table.setItem(row, 6,
                QTableWidgetItem(", ".join(os.path.basename(p) for p, _ in it.file_data)))

            ci = QTableWidgetItem()
            ci.setBackground(QColor(self._color_for(it)))
            ci.setText(self._color_for(it))
            self.data_table.setItem(row, 7, ci)

    # ─────────────────────────────────────────────────────────────────
    # EDIT HELPERS
    # ─────────────────────────────────────────────────────────────────
    def rename_selected_state(self):
        row = self.staging_list.currentRow()
        if row == -1: QMessageBox.warning(self, "No Selection", "Select a state."); return
        self.undo_mgr.push(self.plot_staging_list)
        new_name = self.rename_input.text().strip()
        self.plot_staging_list[row].custom_name = new_name or None
        self._dirty = True; self.sync_sidebar_list(); self.update_plot()

    def color_selected_state(self):
        row = self.staging_list.currentRow()
        if row == -1: QMessageBox.warning(self, "No Selection", "Select a state."); return
        color = QColorDialog.getColor()
        if color.isValid():
            self.undo_mgr.push(self.plot_staging_list)
            self.plot_staging_list[row].custom_color = color.name()
            self._dirty = True; self.sync_sidebar_list(); self.update_plot()

    def move_staging_item(self, direction):
        row = self.staging_list.currentRow()
        tgt = row + direction
        if 0 <= tgt < len(self.plot_staging_list):
            self.undo_mgr.push(self.plot_staging_list)
            self.plot_staging_list[row], self.plot_staging_list[tgt] = \
                self.plot_staging_list[tgt], self.plot_staging_list[row]
            self._dirty = True; self.sync_sidebar_list()
            self.staging_list.setCurrentRow(tgt); self.update_plot()

    def move_table_item(self, direction):
        row = self.data_table.currentRow()
        tgt = row + direction
        if 0 <= tgt < len(self.plot_staging_list):
            self.undo_mgr.push(self.plot_staging_list)
            self.plot_staging_list[row], self.plot_staging_list[tgt] = \
                self.plot_staging_list[tgt], self.plot_staging_list[row]
            self._dirty = True; self.sync_sidebar_list()
            self.update_plot(); self.data_table.setCurrentCell(tgt, 0)

    def remove_staged_items(self):
        rows = sorted({self.staging_list.row(i) for i in self.staging_list.selectedItems()},
                      reverse=True)
        if not rows: return
        if QMessageBox.question(self, "Remove", f"Remove {len(rows)} item(s)?",
                                QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.undo_mgr.push(self.plot_staging_list)
            for r in rows: self.plot_staging_list.pop(r)
            self._dirty = True; self.sync_sidebar_list(); self.update_plot()

    def remove_table_items(self):
        rows = sorted({i.row() for i in self.data_table.selectedIndexes()}, reverse=True)
        if not rows: return
        if QMessageBox.question(self, "Remove", f"Remove {len(rows)} row(s)?",
                                QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.undo_mgr.push(self.plot_staging_list)
            for r in rows: self.plot_staging_list.pop(r)
            self._dirty = True; self.sync_sidebar_list(); self.update_plot()

    def clear_staging(self):
        if self.plot_staging_list and \
                QMessageBox.question(self, "Clear", "Clear all items?",
                                     QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.undo_mgr.push(self.plot_staging_list)
            self.plot_staging_list.clear()
            self.staging_list.clear()
            self._dirty = True; self.update_plot()

    def reset_all_labels(self):
        if QMessageBox.question(self, "Reset", "Reset all label positions?",
                                QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.undo_mgr.push(self.plot_staging_list)
            for it in self.plot_staging_list:
                it.label_offset = [0.0, 0.5]; it.delta_offset = [0.05, 0.0]
            self._dirty = True; self.update_plot()

    def auto_align_labels(self):
        self.undo_mgr.push(self.plot_staging_list)
        for i, it in enumerate(self.plot_staging_list):
            it.label_offset[0] = 0.0
            it.label_offset[1] = 0.5 + (i % 3) * 0.35
        self._dirty = True; self.update_plot()
        self.statusBar().showMessage("Labels auto-aligned", 3000)

    def perform_undo(self):
        res = self.undo_mgr.undo(self.plot_staging_list)
        if res is not None:
            self.plot_staging_list = res
            self.sync_sidebar_list(); self.update_plot()
            self.statusBar().showMessage("Undo", 2000)
        else:
            self.statusBar().showMessage("Nothing to undo", 2000)

    def perform_redo(self):
        res = self.undo_mgr.redo(self.plot_staging_list)
        if res is not None:
            self.plot_staging_list = res
            self.sync_sidebar_list(); self.update_plot()
            self.statusBar().showMessage("Redo", 2000)
        else:
            self.statusBar().showMessage("Nothing to redo", 2000)

    # ─────────────────────────────────────────────────────────────────
    # EXPORT / IMPORT / SESSION
    # ─────────────────────────────────────────────────────────────────
    def copy_to_clipboard(self):
        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=1200, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
        img = QPixmap(); img.loadFromData(buf.getvalue())
        QApplication.clipboard().setPixmap(img)
        self.statusBar().showMessage("✅ Copied 1200 DPI to Clipboard!", 3000)

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graphic", "orca_profile.svg",
            "Vector (*.svg);;PNG (*.png);;PDF (*.pdf);;TIFF (*.tiff)")
        if path:
            dpi = 1200 if not path.endswith('.svg') else 300
            self.fig.savefig(path, dpi=dpi, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            self.statusBar().showMessage(f"✅ Exported to {os.path.basename(path)}", 3000)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "orca_data.csv", "CSV (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([self.data_table.horizontalHeaderItem(c).text()
                                for c in range(self.data_table.columnCount())])
                    for r in range(self.data_table.rowCount()):
                        w.writerow([
                            (self.data_table.item(r, c).text()
                             if self.data_table.item(r, c) else "")
                            for c in range(self.data_table.columnCount())])
                self.statusBar().showMessage(
                    f"✅ Data exported to {os.path.basename(path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv);;All (*)")
        if not path: return
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if not {'Label','Energy_Eh'}.issubset(set(reader.fieldnames or [])):
                    QMessageBox.warning(self, "Invalid CSV",
                        "CSV must have columns: Label, Energy_Eh\n"
                        f"(optional: Pathway, Color)\nFound: {', '.join(reader.fieldnames or [])}"); return
                count = 0
                for row in reader:
                    label  = row.get('Label','').strip()
                    e_str  = row.get('Energy_Eh','').strip()
                    color  = row.get('Color','').strip() or None
                    pw     = row.get('Pathway','').strip() or 'Imported'
                    if not label or not e_str: continue
                    try: e_val = float(e_str)
                    except ValueError: continue
                    self._ensure_pathway_color(pw)
                    it = PlotItem(file_data=[], display_name=label,
                                  folder_path='__imported__', pathway=pw,
                                  energy=e_val, custom_color=color)
                    self.plot_staging_list.append(it)
                    li = QListWidgetItem(f"[{pw}]  {label}")
                    li.setForeground(QColor(color or self.pathway_colors[pw]))
                    self.staging_list.addItem(li)
                    count += 1
            self._dirty = True; self.refresh_pathway_list_widget(); self.update_plot()
            self.statusBar().showMessage(f"✅ Imported {count} entries", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _gather_all_settings(self) -> dict:
        return {
            'title':         self.title_input.text(),
            'y_label':       self.y_label_input.text(),
            'energy_type':   self.energy_type_combo.currentText(),
            'theme':         self.theme_combo.currentText(),
            'legend_loc':    self.legend_loc_combo.currentText(),
            'line_style':    self.line_style_combo.currentText(),
            'zero_ref':      self.zero_ref_combo.currentText(),
            'label_mode':    self.label_mode_combo.currentText(),
            'xlabel_rotation': self.xlabel_rotation_spin.value(),
            'width':         self.fig_width_spin.value(),
            'height':        self.fig_height_spin.value(),
            'state_width':   self.state_width_slider.value(),
            'bar_thickness': self.bar_thickness_spin.value(),
            'margin_left':   self.margin_left_spin.value(),
            'pathway_gap':   self.pathway_gap_spin.value(),
            'base_fs':       self.base_fs_spin.value(),
            'tick_fs':       self.tick_fs_spin.value(),
            'energy_fs':     self.energy_fs_spin.value(),
            'delta_fs':      self.delta_fs_spin.value(),
            'show_legend':   self.check_show_legend.isChecked(),
            'pub_quality':   self.check_pub_quality.isChecked(),
            'show_e_labels': self.check_show_e_labels.isChecked(),
            'show_delta':    self.check_show_delta.isChecked(),
            'auto_align':    self.check_auto_align.isChecked(),
            'interactive':   self.check_interactive.isChecked(),
            'tick_bold':     self.check_tick_bold.isChecked(),
            'energy_bold':   self.check_energy_bold.isChecked(),
            'delta_bold':    self.check_delta_bold.isChecked(),
            'show_xaxis':    self.check_show_xaxis.isChecked(),
            'show_yaxis':    self.check_show_yaxis.isChecked(),
            'show_frame':    self.check_show_frame.isChecked(),
            'inward_ticks':  self.check_inward_ticks.isChecked(),
            'grid':          self.check_grid.isChecked(),
            'smooth':        self.check_smooth.isChecked(),
            'overlay':       self.check_overlay.isChecked(),
            'shadow':        self.check_shadow.isChecked(),
            'smart_labels':  self.check_smart_labels.isChecked(),
            'conn_style':    self.conn_style_combo.currentText(),
            'extra_thermo_display': self.extra_thermo_display_combo.currentText(),
            'label_box':     self.label_box_combo.currentText(),
            'delta_box':     self.delta_box_combo.currentText(),
            'delta_arrow':   self.delta_arrow_combo.currentText(),
            'unit_kcal':     self.btn_kcal.isChecked(),
            'unit_kj':       self.btn_kj.isChecked(),
            'unit_ev':       self.btn_ev.isChecked(),
            'pathway_colors': self.pathway_colors,
            'cross_diffs':   self.cross_diffs,
        }

    def _restore_all_settings(self, s: dict):
        self.title_input.setText(s.get('title','Reaction Energy Profile'))
        self.y_label_input.setText(s.get('y_label','Relative Energy (kcal/mol)'))
        for combo, key, default in [
            (self.energy_type_combo,          'energy_type',          'Gibbs'),
            (self.theme_combo,                'theme',                'Default'),
            (self.legend_loc_combo,           'legend_loc',           'best'),
            (self.line_style_combo,           'line_style',           'Solid (-)'),
            (self.label_mode_combo,           'label_mode',           'Tick Labels Only'),
            (self.conn_style_combo,           'conn_style',           'Smooth (Bezier)'),
            (self.extra_thermo_display_combo, 'extra_thermo_display', 'None'),
            (self.label_box_combo,            'label_box',            'Rounded Box'),
            (self.delta_box_combo,            'delta_box',            'Rounded Box'),
            (self.delta_arrow_combo,          'delta_arrow',          'Straight →'),
        ]:
            idx = combo.findText(s.get(key, default))
            if idx >= 0: combo.setCurrentIndex(idx)
        self.xlabel_rotation_spin.setValue(s.get('xlabel_rotation', 0))
        self.fig_width_spin.setValue(s.get('width',10.0))
        self.fig_height_spin.setValue(s.get('height',7.0))
        self.state_width_slider.setValue(s.get('state_width',0.6))
        self.bar_thickness_spin.setValue(s.get('bar_thickness',3.5))
        self.margin_left_spin.setValue(s.get('margin_left',0.12))
        self.pathway_gap_spin.setValue(s.get('pathway_gap',0.5))
        self.base_fs_spin.setValue(s.get('base_fs',10))
        self.tick_fs_spin.setValue(s.get('tick_fs',10))
        self.energy_fs_spin.setValue(s.get('energy_fs',9))
        self.delta_fs_spin.setValue(s.get('delta_fs',8))
        for attr, key, dflt in [
            ('check_show_legend','show_legend',True),
            ('check_pub_quality','pub_quality',False),
            ('check_show_e_labels','show_e_labels',True),
            ('check_show_delta','show_delta',True),
            ('check_auto_align','auto_align',True),
            ('check_interactive','interactive',True),
            ('check_tick_bold','tick_bold',False),
            ('check_energy_bold','energy_bold',True),
            ('check_delta_bold','delta_bold',True),
            ('check_show_xaxis','show_xaxis',True),
            ('check_show_yaxis','show_yaxis',True),
            ('check_show_frame','show_frame',True),
            ('check_inward_ticks','inward_ticks',True),
            ('check_grid','grid',True),
            ('check_smooth','smooth',True),
            ('check_overlay','overlay',True),
            ('check_shadow','shadow',False),
            ('check_smart_labels','smart_labels',True),
        ]:
            getattr(self, attr).setChecked(s.get(key, dflt))
        self.btn_kcal.setChecked(s.get('unit_kcal',True))
        self.btn_kj.setChecked(s.get('unit_kj',False))
        self.btn_ev.setChecked(s.get('unit_ev',False))
        self.pathway_colors = s.get('pathway_colors', {})
        # Restore cross-pathway diffs
        self.cross_diffs = s.get('cross_diffs', [])
        self.xp_list.clear()
        for diff in self.cross_diffs:
            ia = diff.get('ia', 0); ib = diff.get('ib', 0)
            col = diff.get('color', '#e11d48')
            li  = QListWidgetItem(f"State {ia}  ↔  State {ib}")
            li.setForeground(QColor(col))
            self.xp_list.addItem(li)

    def new_session(self, _skip_prompt=False):
        if not _skip_prompt and self.plot_staging_list:
            if QMessageBox.question(self,"New","Start a new session? Unsaved work will be lost.",
                                    QMessageBox.Yes|QMessageBox.No) == QMessageBox.No:
                return
        self.plot_staging_list.clear()
        self.staging_list.clear()
        self.folders.clear()
        self.pathway_colors.clear()
        self.cross_diffs.clear()
        self.xp_list.clear()
        while self.tab_folders.count(): self.tab_folders.removeTab(0)
        self._dirty = False; self.update_plot()

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self,"Save Session","","JSON (*.json)")
        if path:
            try:
                with open(path,'w') as f:
                    json.dump({'version': 4,
                               'staging': [it.to_dict() for it in self.plot_staging_list],
                               'folders': list(self.folders.keys()),
                               'settings': self._gather_all_settings()},
                              f, indent=4, default=str)
                self._dirty = False
                self.statusBar().showMessage(f"✅ Saved to {os.path.basename(path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self,"Save Error", str(e))

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self,"Open Session","","JSON (*.json)")
        if not path: return
        try:
            with open(path,'r') as f: data = json.load(f)
            self.new_session(_skip_prompt=True)
            for d in data.get('staging', []):
                for key in ('label_offset','delta_offset'):
                    v = d.get(key)
                    if not isinstance(v, list):
                        try:
                            v = json.loads(v)
                        except Exception:
                            v = [0.0, 0.5] if key=='label_offset' else [0.05, 0.0]
                    d[key] = v if isinstance(v, list) else \
                             ([0.0, 0.5] if key=='label_offset' else [0.05, 0.0])
                # Ensure all fields exist (backward compat with v2/v3)
                d.setdefault('pathway',          '')
                d.setdefault('show_extra_thermo', False)
                d.setdefault('extra_thermo_type', 'Enthalpy')
                d.setdefault('conn_arrow_style',  'Default')
                # Drop any unknown keys that old versions might have stored
                known = {f.name for f in PlotItem.__dataclass_fields__.values()}
                d = {k: v for k, v in d.items() if k in known}
                self.plot_staging_list.append(PlotItem(**d))
            for fp in data.get('folders', []):
                if os.path.exists(fp):
                    loader = FolderLoader(fp)
                    self._active_loaders.append(loader)
                    loader.finished.connect(self.on_folder_loaded)
                    loader.finished.connect(lambda _d, l=loader: self._release_loader(l))
                    loader.error.connect(lambda _e, l=loader:   self._release_loader(l))
                    loader.start()
            self._restore_all_settings(data.get('settings',{}))
            self.sync_sidebar_list()
            self._dirty = False; self.update_plot()
            self.statusBar().showMessage(f"✅ Loaded {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self,"Load Error", str(e))

    # ─────────────────────────────────────────────────────────────────
    # PRESETS & STYLES
    # ─────────────────────────────────────────────────────────────────
    def apply_journal_preset(self, _index=None):
        p = self.journal_preset_combo.currentText()
        if p == "Custom": return
        self.check_pub_quality.setChecked(True)
        self.check_inward_ticks.setChecked(True)
        self.check_grid.setChecked(False)
        self.margin_left_spin.setValue(0.15)
        self.check_shadow.setChecked(False)
        presets = {
            "JACS (Single Column)":  (3.25, 2.7,  8,  8, 7, 7),
            "JACS (Double Column)":  (7.0,  5.0, 10, 10, 9, 9),
            "Nature/Science":        (3.5,  3.0,  7,  7, 6, 6),
            "Angewandte":            (8.5,  6.0,  9,  9, 8, 8),
            "ACS Catalysis":         (3.5,  2.8,  8,  8, 7, 7),
        }
        for key, (w, h, b, t, e, d) in presets.items():
            if key in p:
                self.fig_width_spin.setValue(w); self.fig_height_spin.setValue(h)
                self.base_fs_spin.setValue(b);   self.tick_fs_spin.setValue(t)
                self.energy_fs_spin.setValue(e); self.delta_fs_spin.setValue(d)
                break
        self.update_plot()
        self.statusBar().showMessage(f"Applied {p} preset", 3000)

    def apply_quick_style(self, style_name):
        if style_name == "Dark":
            self.theme_combo.setCurrentText("dark_background")
        elif style_name == "Light":
            self.theme_combo.setCurrentText("Default")
        elif style_name == "Colorful":
            self.theme_combo.setCurrentText("seaborn-v0_8")
        self.update_plot()
        self.statusBar().showMessage(f"Applied {style_name} style", 3000)

    def closeEvent(self, event):
        # Close the floating control window with the main window
        self.float_ctrl_win.close()
        # Stop any background loaders before closing
        for loader in list(self._active_loaders):
            loader.stop()
            loader.wait(2000)   # max 2 s

        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Quit anyway?",
                QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
            if reply == QMessageBox.Save:
                self.save_session(); event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def show_about(self):
        QMessageBox.about(self, "About ORCA Professional Plotter v6.0", """
<h2>ORCA Professional Plotter v6.0</h2>
<p><b>Multi-Pathway Edition</b> — advanced energy profile visualisation for quantum chemistry.</p>
<p><b>What's new in v6.0:</b></p>
<ul>
  <li><b>★ Named Pathway System:</b> group any states into a named cascade, even from the same folder.
      Use <i>Add as Pathway…</i> or the Pathway Manager panel.</li>
  <li><b>★ Pathway Gap control:</b> add horizontal space between cascades.</li>
  <li><b>★ State Label Modes:</b> Tick Labels / Inline Below / Inline Above / Both / None.</li>
  <li><b>★ X-label rotation:</b> 0°–90° for long state names.</li>
  <li><b>★ Smart Overlap Avoidance:</b> energy value labels shift automatically to avoid collision.</li>
  <li><b>★ Per-pathway zero reference:</b> each cascade referenced to its own first state.</li>
  <li><b>★ Pathway colours:</b> individually controllable, saved in session.</li>
  <li><b>★ Improved Δ annotations:</b> boxed labels with clean arrows.</li>
  <li>Undo / Redo (Ctrl+Z / Ctrl+Y), 1200 DPI clipboard, CSV import / export, full session save.</li>
</ul>
<p>© 2024–2026 ORCA Plotter Development Team</p>
""")


# ==================== Entry Point ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        * { color:#111111; font-family:"Segoe UI",Arial,sans-serif; }
        QMainWindow { background-color:#f0f0f0; }
        QPushButton { padding:8px 16px; border:none; border-radius:8px;
            background-color:#2563eb; color:#ffffff; font-weight:600; font-size:12px; }
        QPushButton:hover    { background-color:#1d4ed8; }
        QPushButton:pressed  { background-color:#1e3a8a; }
        QPushButton:disabled { background-color:#9ca3af; color:#d1d5db; }
        QLabel { color:#111111; font-size:12px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            padding:8px; border:1px solid #9ca3af; border-radius:8px;
            background-color:#ffffff; color:#111111; font-size:12px; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border:2px solid #2563eb; }
        QComboBox QAbstractItemView { background:#ffffff; color:#111111;
            selection-background-color:#2563eb; selection-color:#ffffff; }
        QCheckBox { spacing:8px; font-size:12px; color:#111111; }
        QCheckBox::indicator { width:18px; height:18px; }
        QGroupBox { font-weight:bold; border:1px solid #9ca3af; border-radius:12px;
            margin-top:12px; padding-top:16px; background-color:#ffffff;
            color:#111111; font-size:13px; }
        QGroupBox::title { subcontrol-origin:margin; left:16px; padding:0 8px; color:#000; }
        QToolBar { background-color:#e5e7eb; border-bottom:2px solid #9ca3af; spacing:4px; }
        QToolBar QToolButton { color:#111111; font-weight:500; padding:4px 8px; }
        QToolBar QToolButton:hover { background-color:#d1d5db; border-radius:4px; }
        QStatusBar { background-color:#e5e7eb; color:#111111; font-size:11px; font-weight:500; }
        QTabWidget::pane { border:2px solid #9ca3af; background-color:#ffffff; }
        QTabBar::tab { color:#111111; background-color:#e5e7eb; border:1px solid #9ca3af;
            padding:8px 16px; font-weight:bold; }
        QTabBar::tab:selected { background-color:#ffffff; color:#1e3a8a; border-bottom-color:#fff; }
        QListWidget { color:#111111; background-color:#ffffff; border:1px solid #9ca3af; }
        QListWidget::item { color:#111111; }
        QListWidget::item:selected { background-color:#2563eb; color:#ffffff; }
        QTableWidget { color:#111111; background-color:#ffffff; gridline-color:#9ca3af; }
        QHeaderView::section { background-color:#e5e7eb; color:#111111;
            font-weight:bold; border:1px solid #9ca3af; padding:6px; }
        QTableWidget::item:selected { background-color:#2563eb; color:#ffffff; }
        QScrollBar:vertical { border:none; background:#e5e7eb; width:12px; }
        QScrollBar::handle:vertical { background:#6b7280; border-radius:6px; min-height:30px; }
        QMenuBar { background-color:#e5e7eb; color:#111111; }
        QMenuBar::item:selected { background-color:#2563eb; color:#ffffff; }
        QMenu { background-color:#ffffff; color:#111111; border:1px solid #9ca3af; }
        QMenu::item:selected { background-color:#2563eb; color:#ffffff; }
    """)
    window = OrcaAnalyzerApp()
    window.show()
    sys.exit(app.exec())
