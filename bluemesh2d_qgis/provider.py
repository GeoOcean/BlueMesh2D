"""Processing provider that exposes the BlueMesh2D algorithms."""

from qgis.core import QgsProcessingProvider

from .algorithm import ALL_ALGORITHMS, plugin_icon


class BlueMesh2DProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        for alg in ALL_ALGORITHMS:
            self.addAlgorithm(alg())

    def icon(self):
        return plugin_icon()

    def id(self):
        return "bluemesh2d"

    def name(self):
        return "BlueMesh2D"

    def longName(self):
        return "BlueMesh2D — unstructured mesh generation"
