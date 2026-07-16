"""Plugin bootstrap: registers the BlueMesh2D Processing provider with QGIS.

Heavy imports (numpy/shapely/rasterio via the algorithms module) are deferred
to :meth:`BlueMesh2DPlugin.initGui`, behind a dependency check: when packages
are missing the plugin loads "empty" and shows an install dialog instead of
crashing QGIS's plugin loader with a raw traceback.
"""

from qgis.core import QgsApplication

from .deps_installer import REQUIRED, DepsDialog, activate_venv, find_missing


class BlueMesh2DPlugin:
    """QGIS plugin entry point: registers the BlueMesh2D Processing provider.

    Parameters
    ----------
    iface : qgis.gui.QgisInterface
        QGIS application interface, passed in by :func:`classFactory`.
    """

    MENU = "&BlueMesh2D"

    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.deps_action = None

    def _main_window(self):
        try:
            return self.iface.mainWindow()
        except Exception:
            return None  # headless (qgis_process): no GUI available

    def initProcessing(self):
        # packages installed into the plugin-managed venv (PEP 668 systems)
        # must be on sys.path before checking what is missing
        activate_venv()
        missing = find_missing(REQUIRED)
        if missing:
            msg = ("Missing Python packages: " + ", ".join(missing)
                   + " — use Plugins > BlueMesh2D > Check / install "
                   "dependencies, then restart QGIS.")
            parent = self._main_window()
            if parent is not None:
                try:
                    self.iface.messageBar().pushWarning("BlueMesh2D", msg)
                except Exception:
                    pass
                # defer until QGIS finishes starting up: opening the modal
                # dialog inside the plugin-load phase blocks startup and
                # keeps QGIS's busy cursor spinning over the dialog
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(0, lambda: DepsDialog(parent).exec())
            else:
                import sys
                sys.stderr.write("BlueMesh2D: " + msg + "\n")
            # Even when the install just succeeded, DO NOT continue in this
            # session: importing freshly installed binary wheels (rasterio,
            # shapely) next to QGIS's own GDAL/GEOS can crash the process.
            # The algorithms appear after a QGIS restart.
            return
        try:
            from .provider import BlueMesh2DProvider
        except ModuleNotFoundError as exc:
            # dependencies are fine, so this is an internal packaging problem
            # (e.g. a stale bundled `bluemesh2d` copy), not a pip issue
            msg = ("BlueMesh2D failed to load because of a plugin packaging "
                   f"problem (not missing dependencies):\n\n{exc}\n\n"
                   "Reinstall the plugin (the bundled 'bluemesh2d' package "
                   "looks incomplete or out of date).")
            parent = self._main_window()
            if parent is not None:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(parent, "BlueMesh2D", msg)
            else:
                import sys
                sys.stderr.write("BlueMesh2D: " + msg + "\n")
            return
        self.provider = BlueMesh2DProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        from qgis.PyQt.QtWidgets import QAction

        self.deps_action = QAction("Check / install dependencies…",
                                   self.iface.mainWindow())
        self.deps_action.triggered.connect(self._show_deps_dialog)
        self.iface.addPluginToMenu(self.MENU, self.deps_action)

        self.initProcessing()

    def _show_deps_dialog(self):
        DepsDialog(self.iface.mainWindow()).exec()

    def unload(self):
        if self.deps_action is not None:
            self.iface.removePluginMenu(self.MENU, self.deps_action)
            self.deps_action = None
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
