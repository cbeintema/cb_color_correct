from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def apply_ableton_theme(app: QtWidgets.QApplication) -> None:
    """Apply an Ableton-ish dark theme using Fusion + palette + QSS.

    Stays within standard Qt widgets (no custom painting).
    """

    app.setStyle("Fusion")

    pal = QtGui.QPalette()

    # Core surfaces
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(24, 24, 24))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(18, 18, 18))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(24, 24, 24))

    # Text
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(232, 232, 232))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(232, 232, 232))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(232, 232, 232))

    # Buttons
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(34, 34, 34))

    # Selection / accent
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(95, 185, 110))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))

    # Tooltips
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(232, 232, 232))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(24, 24, 24))

    app.setPalette(pal)

    app.setStyleSheet(
        """
        /* General */
        QWidget { background: #181818; color: #e8e8e8; }
        QMainWindow { background: #181818; }

        /* Labels */
        QLabel { background: transparent; }

        /* Buttons */
        QPushButton, QToolButton {
            background: #222222;
            border: 1px solid #343434;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover, QToolButton:hover { background: #2a2a2a; border-color: #3c3c3c; }
        QPushButton:pressed, QToolButton:pressed { background: #1f1f1f; }
        QPushButton:disabled, QToolButton:disabled { color: #8a8a8a; background: #1a1a1a; border-color: #2a2a2a; }

        /* Combos */
        QComboBox {
            background: #1f1f1f;
            border: 1px solid #343434;
            border-radius: 4px;
            padding: 2px 8px;
            min-height: 22px;
        }
        QComboBox:hover { border-color: #3c3c3c; }
        QComboBox::drop-down { border: 0px; width: 18px; }

        /* Line edits */
        QLineEdit {
            background: #121212;
            border: 1px solid #343434;
            border-radius: 4px;
            padding: 3px 8px;
            min-height: 22px;
        }
        QLineEdit:hover { border-color: #3c3c3c; }
        QLineEdit:focus { border-color: #5fb96e; }

        /* Trees / lists */
        QTreeWidget {
            background: #121212;
            border: 1px solid #343434;
            border-radius: 4px;
            outline: none;
        }
        QTreeWidget::item {
            padding: 4px 6px;
            height: 22px;
            min-height: 22px;
            max-height: 22px;
        }
        QTreeWidget::item:selected { background: #294833; }
        QTreeWidget::item:hover { background: #1d1d1d; }
        QTreeWidget::item:selected:!active { background: #294833; }

        /* Group boxes: flatter, Ableton-ish */
        QGroupBox {
            background: #171717;
            border: 1px solid #2e2e2e;
            border-radius: 4px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #cfcfcf;
        }
        QGroupBox::indicator { width: 14px; height: 14px; }

        /* Sliders */
        QSlider::groove:horizontal { height: 4px; background: #3a3a3a; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #5fb96e; border-radius: 2px; }
        QSlider::add-page:horizontal { background: #2a2a2a; border-radius: 2px; }
        QSlider::handle:horizontal { width: 12px; margin: -6px 0; border-radius: 6px; background: #d0d0d0; }

        /* Scrollbars */
        QScrollBar:vertical {
            background: #111111;
            width: 10px;
            margin: 0px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background: #2c2c2c;
            min-height: 24px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover { background: #3a3a3a; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

        QScrollBar:horizontal {
            background: #111111;
            height: 10px;
            margin: 0px;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background: #2c2c2c;
            min-width: 24px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover { background: #3a3a3a; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

        /* Dialogs */
        QMessageBox { background: #181818; }

        /* Progress */
        QProgressBar {
            background: #121212;
            border: 1px solid #343434;
            border-radius: 4px;
            text-align: center;
            padding: 1px;
            color: #cfcfcf;
            min-height: 18px;
        }
        QProgressBar::chunk {
            background-color: #5fb96e;
            border-radius: 3px;
        }
        """
    )
