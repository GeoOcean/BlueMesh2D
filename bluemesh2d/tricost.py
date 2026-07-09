import matplotlib.pyplot as plt
import numpy as np

from .mesh_cost.relhfn import relhfn
from .mesh_cost.triang import triang
from .mesh_cost.trideg import trideg
from .mesh_cost.triscr import triscr


def tricost(vert=None, conn=None, tria=None, tnum=None, hvrt=None):
    """Plot quality-metric histograms for a 2D triangle mesh.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    conn : ndarray of shape (E, 2)
        Constrained edges.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    tnum : ndarray of shape (T, 1), optional
        Part index per triangle.
    hvrt : ndarray of shape (V,), optional
        Vertex mesh-size values; when provided, also plots relative edge lengths.

    Notes
    -----
    Displays histograms of element quality scores, angles, relative sizes
    (when ``hvrt`` is given), and node degree.

    References
    ----------
    Translation of the MESH2D function ``TRICOST``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not all(
        isinstance(x, (np.ndarray, type(None))) for x in [vert, conn, tria, tnum, hvrt]
    ):
        raise TypeError("tricost:incorrectInputClass - Incorrect input class.")

    if vert is None or conn is None or tria is None:
        raise ValueError("tricost: missing required inputs (vert, conn, tria).")

    if vert.ndim != 2 or conn.ndim != 2 or tria.ndim != 2:
        raise ValueError("tricost:incorrectDimensions - Inputs must be 2D arrays.")

    if vert.shape[1] != 2 or conn.shape[1] < 2 or tria.shape[1] < 3:
        raise ValueError("tricost:incorrectDimensions - Invalid input sizes.")

    nvrt = vert.shape[0]
    _ntri = tria.shape[0]

    if np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) > nvrt:
        raise ValueError("tricost:invalidInputs - Invalid EDGE input array.")

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) > nvrt:
        raise ValueError("tricost:invalidInputs - Invalid TRIA input array.")

    axpos31 = [0.125, 0.750, 0.800, 0.150]
    axpos32 = [0.125, 0.450, 0.800, 0.150]
    axpos33 = [0.125, 0.150, 0.800, 0.150]

    axpos41 = [0.125, 0.835, 0.800, 0.135]
    axpos42 = [0.125, 0.590, 0.800, 0.135]
    axpos43 = [0.125, 0.345, 0.800, 0.135]
    axpos44 = [0.125, 0.100, 0.800, 0.135]

    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor("w")

    if hvrt is not None and hvrt.size > 0:
        ax1 = fig.add_axes(axpos41)
        ax1.set_title("Quality score")
        scrhist(triscr(vert, tria), "tria3", ax=ax1)

        ax2 = fig.add_axes(axpos42)
        ax2.set_title("Angles")
        anghist(triang(vert, tria), "tria3", ax=ax2)

        ax3 = fig.add_axes(axpos43)
        ax3.set_title("Relative size")
        hfnhist(relhfn(vert, tria, hvrt), "tria3", ax=ax3)

        ax4 = fig.add_axes(axpos44)
        ax4.set_title("Node degree")
        deghist(trideg(vert, tria), "tria3", ax=ax4)

    else:
        ax1 = fig.add_axes(axpos31)
        ax1.set_title("Quality score")
        scrhist(triscr(vert, tria), "tria3", ax=ax1)

        ax2 = fig.add_axes(axpos32)
        ax2.set_title("Angles")
        anghist(triang(vert, tria), "tria3", ax=ax2)

        ax3 = fig.add_axes(axpos33)
        ax3.set_title("Node degree")
        deghist(trideg(vert, tria), "tria3", ax=ax3)


def mad(ff):
    """Compute the mean absolute deviation from the mean.

    Parameters
    ----------
    ff : array_like
        Input values.

    Returns
    -------
    mf : float
        Mean absolute deviation.

    References
    ----------
    Translation of the MESH2D function ``MAD``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    ff = np.asarray(ff, dtype=float)
    mf = np.mean(np.abs(ff - np.mean(ff)))
    return mf


def deghist(dd, ty, ax=None):
    """Plot a histogram of node degree.

    Parameters
    ----------
    dd : array_like
        Node degrees.
    ty : str
        Mesh type label (``'tria3'`` or ``'tria4'``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.

    References
    ----------
    Translation of the MESH2D function ``DEGHIST``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    if ax is None:
        ax = plt.gca()
    dd = np.asarray(dd).ravel()
    be = np.arange(1, np.max(dd) + 1)
    hc, _ = np.histogram(dd, bins=np.append(be, np.max(be) + 1))

    k = (0.60, 0.60, 0.60)

    # Histogram
    ax.bar(be, hc, width=1.05, color=k, edgecolor=k)

    ax.axis("tight")
    ax.set_yticks([])
    ax.set_xticks(np.arange(2, 13, 2))
    ax.tick_params(axis="both", which="both", length=5, width=2)
    ax.set_xlim([0, 12])
    ax.set_xlabel("", fontsize=22)

    # Axis label depends on the triangulation type
    if ty == "tria4":
        ax.text(-0.225, 0, r"$|d|_{\tau}$", ha="right", fontsize=22)
    elif ty == "tria3":
        ax.text(-0.225, 0, r"$|d|_{f}$", ha="right", fontsize=22)


def anghist(ad, ty, ax=None):
    """Plot a histogram of triangle angles.

    Parameters
    ----------
    ad : array_like
        Internal angles in degrees.
    ty : str
        Mesh type label (``'tria3'`` or ``'tria4'``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.

    References
    ----------
    Translation of the MESH2D function ``ANGHIST``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    if ax is None:
        ax = plt.gca()
    ad = np.asarray(ad).ravel()
    be = np.linspace(0.0, 180.0, 91)
    bm = (be[:-1] + be[1:]) / 2.0
    hc, _ = np.histogram(ad, bins=be)

    if ty == "tria4":
        poor = (bm < 10.0) | (bm >= 160.0)
        okay = ((bm >= 10.0) & (bm < 20.0)) | ((bm >= 140.0) & (bm < 160.0))
        good = ((bm >= 20.0) & (bm < 30.0)) | ((bm >= 120.0) & (bm < 140.0))
        best = (bm >= 30.0) & (bm < 120.0)
    elif ty == "tria3":
        poor = (bm < 15.0) | (bm >= 150.0)
        okay = ((bm >= 15.0) & (bm < 30.0)) | ((bm >= 120.0) & (bm < 150.0))
        good = ((bm >= 30.0) & (bm < 45.0)) | ((bm >= 90.0) & (bm < 120.0))
        best = (bm >= 45.0) & (bm < 90.0)
    else:
        raise ValueError("Type must be 'tria3' or 'tria4'.")

    r = (0.85, 0.00, 0.00)
    y = (1.00, 0.95, 0.00)
    g = (0.00, 0.90, 0.00)
    k = (0.60, 0.60, 0.60)

    ax.bar(bm[poor], hc[poor], width=1.05, color=r, edgecolor=r)
    ax.bar(bm[okay], hc[okay], width=1.05, color=y, edgecolor=y)
    ax.bar(bm[good], hc[good], width=1.05, color=g, edgecolor=g)
    ax.bar(bm[best], hc[best], width=1.05, color=k, edgecolor=k)

    ax.axis("tight")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlim([0.0, 180.0])
    ax.tick_params(axis="both", which="both", length=5, width=2, labelsize=14)

    mina = max(1.000, np.min(ad))
    maxa = min(179.0, np.max(ad))

    mada_val = mad(ad)

    ax.axvline(x=mina, color="r", linewidth=1.5)
    ax.axvline(x=maxa, color="r", linewidth=1.5)

    if mina > 25.0:
        ax.text(
            mina - 1.8, 0.90 * np.max(hc), f"{np.min(ad):.1f}", ha="right", fontsize=15
        )
    else:
        ax.text(
            mina + 1.8, 0.90 * np.max(hc), f"{np.min(ad):.1f}", ha="left", fontsize=15
        )

    if maxa < 140.0:
        ax.text(
            maxa + 1.8, 0.90 * np.max(hc), f"{np.max(ad):.1f}", ha="left", fontsize=15
        )
    else:
        ax.text(
            maxa - 1.8, 0.90 * np.max(hc), f"{np.max(ad):.1f}", ha="right", fontsize=15
        )

    if maxa < 100.0:
        ax.text(
            maxa - 16.0,
            0.45 * np.max(hc),
            r"$\bar{\sigma}_{\theta}\!=$",
            ha="left",
            fontsize=16,
        )
        ax.text(
            maxa + 1.8, 0.45 * np.max(hc), f"{mada_val:.2f}", ha="left", fontsize=15
        )
    else:
        ax.text(
            maxa - 16.0,
            0.45 * np.max(hc),
            r"$\bar{\sigma}_{\theta}\!=$",
            ha="left",
            fontsize=16,
        )
        ax.text(
            maxa + 1.8, 0.45 * np.max(hc), f"{mada_val:.3f}", ha="left", fontsize=15
        )

    if ty == "tria4":
        ax.text(-9.0, 0.0, r"$\theta_{\tau}$", ha="right", fontsize=22)
    elif ty == "tria3":
        ax.text(-9.0, 0.0, r"$\theta_{f}$", ha="right", fontsize=22)


def scrhist(sc, ty, ax=None):
    """Plot a histogram of element quality scores.

    Parameters
    ----------
    sc : array_like
        Quality scores in ``[0, 1]``.
    ty : str
        Mesh type label (``'tria3'``, ``'tria4'``, ``'dual3'``, or ``'dual4'``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.

    References
    ----------
    Translation of the MESH2D function ``SCRHIST``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if ax is None:
        ax = plt.gca()
    sc = np.asarray(sc).ravel()
    be = np.linspace(0.0, 1.0, 101)
    bm = (be[:-1] + be[1:]) / 2.0
    hc, _ = np.histogram(sc, bins=be)

    if ty in ("tria4", "dual4"):
        poor = bm < 0.25
        okay = (bm >= 0.25) & (bm < 0.50)
        good = (bm >= 0.50) & (bm < 0.75)
        best = bm >= 0.75
    elif ty in ("tria3", "dual3"):
        poor = bm < 0.30
        okay = (bm >= 0.30) & (bm < 0.60)
        good = (bm >= 0.60) & (bm < 0.90)
        best = bm >= 0.90
    else:
        raise ValueError("Type must be 'tria3','tria4','dual3' or 'dual4'.")

    r = (0.85, 0.00, 0.00)
    y = (1.00, 0.95, 0.00)
    g = (0.00, 0.90, 0.00)
    k = (0.60, 0.60, 0.60)

    ax.bar(bm[poor], hc[poor], width=1.05 / 100, color=r, edgecolor=r)
    ax.bar(bm[okay], hc[okay], width=1.05 / 100, color=y, edgecolor=y)
    ax.bar(bm[good], hc[good], width=1.05 / 100, color=g, edgecolor=g)
    ax.bar(bm[best], hc[best], width=1.05 / 100, color=k, edgecolor=k)

    ax.axis("tight")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlim([0.0, 1.0])
    ax.tick_params(axis="both", which="both", length=5, width=2, labelsize=14)

    mins = max(0.010, np.min(sc))
    _maxs = min(0.990, np.max(sc))

    ax.axvline(x=mins, color="r", linewidth=1.5)
    ax.axvline(x=np.mean(sc), color="r", linewidth=1.5)

    if mins > 0.4:
        ax.text(
            mins - 0.01, 0.9 * np.max(hc), f"{np.min(sc):.3f}", ha="right", fontsize=15
        )
    else:
        ax.text(
            mins + 0.01, 0.9 * np.max(hc), f"{np.min(sc):.3f}", ha="left", fontsize=15
        )

    if np.mean(sc) > mins + 0.150:
        ax.text(
            np.mean(sc) - 0.01,
            0.9 * np.max(hc),
            f"{np.mean(sc):.3f}",
            ha="right",
            fontsize=15,
        )

    if ty == "tria4":
        ax.text(
            -0.04, 0.0, r"$\mathcal{Q}^{\mathcal{T}}_{\tau}$", ha="right", fontsize=22
        )
    elif ty == "tria3":
        ax.text(-0.04, 0.0, r"$\mathcal{Q}^{\mathcal{T}}_{f}$", ha="right", fontsize=22)
    elif ty == "dual4":
        ax.text(
            -0.04, 0.0, r"$\mathcal{Q}^{\mathcal{D}}_{\tau}$", ha="right", fontsize=22
        )
    elif ty == "dual3":
        ax.text(-0.04, 0.0, r"$\mathcal{Q}^{\mathcal{D}}_{f}$", ha="right", fontsize=22)


def hfnhist(hf, ty, ax=None):
    """Plot a histogram of relative mesh-size ratios.

    Parameters
    ----------
    hf : array_like
        Ratio of actual to target edge length.
    ty : str
        Mesh type label (kept for API compatibility).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.

    References
    ----------
    Translation of the MESH2D function ``HFNHIST``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if ax is None:
        ax = plt.gca()
    hf = np.asarray(hf).ravel()

    be = np.linspace(0.0, 2.0, 101)
    bm = (be[:-1] + be[1:]) / 2.0
    hc, _ = np.histogram(hf, bins=be)

    poor = (bm < 0.40) | (bm >= 1.6)
    okay = ((bm >= 0.40) & (bm < 0.60)) | ((bm >= 1.4) & (bm < 1.6))
    good = ((bm >= 0.60) & (bm < 0.80)) | ((bm >= 1.2) & (bm < 1.4))
    best = (bm >= 0.80) & (bm < 1.2)

    r = (0.85, 0.00, 0.00)
    y = (1.00, 0.95, 0.00)
    g = (0.00, 0.90, 0.00)
    k = (0.60, 0.60, 0.60)

    ax.bar(bm[poor], hc[poor], width=1.05 / 100, color=r, edgecolor=r)
    ax.bar(bm[okay], hc[okay], width=1.05 / 100, color=y, edgecolor=y)
    ax.bar(bm[good], hc[good], width=1.05 / 100, color=g, edgecolor=g)
    ax.bar(bm[best], hc[best], width=1.05 / 100, color=k, edgecolor=k)

    ax.axis("tight")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, 2.1, 0.5))
    ax.set_xlim([0.0, 2.0])
    ax.tick_params(axis="both", which="both", length=5, width=2, labelsize=14)

    maxhf = np.max(hf)
    ax.axvline(x=maxhf, color="r", linewidth=1.5)

    ax.text(maxhf + 0.02, 0.90 * np.max(hc), f"{maxhf:.2f}", ha="left", fontsize=15)

    ax.text(
        maxhf - 0.18, 0.45 * np.max(hc), r"$\bar{\sigma}_{h} =$", ha="left", fontsize=16
    )
    ax.text(maxhf + 0.02, 0.45 * np.max(hc), f"{mad(hf):.2f}", ha="left", fontsize=15)

    ax.text(-0.10, 0.0, r"$h_{r}$", ha="right", fontsize=22)
