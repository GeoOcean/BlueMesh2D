"""Plugin bootstrap: registers the BlueMesh2D Processing provider with QGIS."""

from qgis.core import QgsApplication

from .provider import BlueMesh2DProvider


class BlueMesh2DPlugin:
    """QGIS plugin entry point: registers the BlueMesh2D Processing provider.

    Parameters
    ----------
    iface : qgis.gui.QgisInterface
        QGIS application interface, passed in by :func:`classFactory`.
    """

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = BlueMesh2DProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
