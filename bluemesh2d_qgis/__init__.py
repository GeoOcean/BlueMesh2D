"""BlueMesh2D QGIS plugin package.

QGIS calls :func:`classFactory` to instantiate the plugin.
"""


def classFactory(iface):
    """Instantiate the plugin (QGIS's mandated plugin entry point).

    Parameters
    ----------
    iface : qgis.gui.QgisInterface
        QGIS application interface, passed in by QGIS.

    Returns
    -------
    plugin : plugin.BlueMesh2DPlugin
        The plugin instance.
    """
    from .plugin import BlueMesh2DPlugin
    return BlueMesh2DPlugin(iface)
