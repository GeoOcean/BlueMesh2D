import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from bluemesh2d.hfun_util.trihfn import trihfn
from bluemesh2d.hfun_util.lfshfn import lfshfn
from bluemesh2d.mesh_util.tridiv import tridiv
from bluemesh2d.mesh_util.idxtri import idxtri
from bluemesh2d.smooth import smooth
from bluemesh2d.refine import refine
from bluemesh2d.triread import triread
from bluemesh2d.tricost import tricost


def tridemo(demo):
    """
    Run various 2D triangulation demo problems from MESH2D.

    This function executes one of the available demonstration problems
    for mesh generation using Delaunay-based techniques.

    Parameters
    ----------
    demo : int
        Index of the demo to run.

    Available Demos
    ---------------
    - DEMO-0 : Very simple example to start with — construct a mesh for
      a square domain with a square hole cut from its center.
    - DEMO-1 : Explore the impact of the "radius-edge" threshold (RHO2)
      on mesh density and quality.
    - DEMO-2 : Compare the "Frontal-Delaunay" and "Delaunay-refinement"
      algorithms.
    - DEMO-3 : Explore the impact of user-defined mesh-size constraints.
    - DEMO-4 : Explore the effects of "hill-climbing" mesh optimization.
    - DEMO-5 : Assemble triangulations for multi-part geometries.
    - DEMO-6 : Assemble triangulations for geometries with internal
      constraints.
    - DEMO-7 : Investigate the use of quadtree-type refinement.
    - DEMO-8 : Explore user-defined mesh-size constraints (variant).
    - DEMO-9 : Large-scale problem: mesh refinement and optimization.
    - DEMO10 : Medium-scale problem: mesh refinement and optimization.

    See Also
    --------
    refine2 : Delaunay-based mesh refinement.
    smooth2 : Smoothing and optimization of triangulations.
    tridiv2 : Triangulation division for multi-part domains.
    fixgeo2 : Geometry correction and boundary preparation.

    References
    ----------
    Translation of the MESH2D function `TRIDEMO` by Darren Engwirda.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Notes
    -----
    Original author: Darren Engwirda
    Email: d.engwirda@gmail.com
    Last updated: 09/07/2018
    """

    if demo == 0:
        demo0()
    elif demo == 1:
        demo1()
    elif demo == 2:
        demo2()
    elif demo == 3:
        demo3()
    elif demo == 4:
        demo4()
    elif demo == 5:
        demo5()
    elif demo == 6:
        demo6()
    elif demo == 7:
        demo7()
    elif demo == 8:
        demo8()
    elif demo == 9:
        demo9()
    elif demo == 10:
        demo10()
    else:
        raise ValueError("tridemo:invalidSelection - Invalid selection!")


def demo0():
    """
    DEMO0 a very simple example to start with -- mesh a square domain with a square hold cut from its centre.
    """

    print(
        " A very simple example to start with -- construct a mesh for \n"
        " a simple square domain with a square hole cut from its cen- \n"
        " tre. The geometry is specified as a Planar Straight-Line \n"
        " Graph (PSLG) -- a list of xy coordinates, or 'nodes', and a \n"
        " list of straight-line connections between nodes, or 'edges'.\n"
        " The refine routine is used to build a triangulation of the \n"
        " domain that: (a) conforms to the geometry, and (b) contains \n"
        " only 'nicely' shaped triangles. In the second panel, a mesh \n"
        " that additionally satisfies 'mesh-size' constrains is cons- \n"
        " structed -- "
    )

    # ------------------------------------------- setup geometry
    node = np.array(
        [  # list of xy "node" coordinates
            [0, 0],  # outer square
            [9, 0],
            [9, 9],
            [0, 9],
            [4, 4],  # inner square
            [5, 4],
            [5, 5],
            [4, 5],
        ]
    )

    edge = (
        np.array(
            [  # list of "edges" between nodes
                [1, 2],  # outer square
                [2, 3],
                [3, 4],
                [4, 1],
                [5, 6],  # inner square
                [6, 7],
                [7, 8],
                [8, 5],
            ]
        )
        - 1
    )
    opts = {}
    # ------------------------------------------- call mesh-gen.
    vert, etri, tria, tnum = refine(node, edge, [], opts)

    # ------------------------------------------- draw tria-mesh
    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    # ------------------------------------------- call mesh-gen with hfun
    hfun = 0.5  # uniform "target" edge-lengths
    vert, etri, tria, tnum = refine(node, edge, [], opts, hfun)

    # ------------------------------------------- draw tria-mesh
    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.draw()
    plt.show()


def demo1():
    """
    DEMO1 explore impact of RHO2 threshold on mesh density/quality
    """

    filepath = os.path.dirname(os.path.abspath(__file__))
    meshfile = os.path.join(filepath, "poly_data", "lake.msh")

    node, edge, _, _ = triread(meshfile)

    print(
        " The refine routine can be used to build guaranteed-quality \n"
        " Delaunay triangulations for general polygonal geometries in \n"
        " the two-dimensional plane. The 'quality' of elements in the \n"
        " triangulation can be controlled using the 'radius-edge' bo- \n"
        " und RHO2. \n"
    )

    # ---------------------------------------------- RHO2 = +1.50
    print("\n")
    print(
        " Setting large values for RHO2, (RHO2 = 1.50 here) generates \n"
        " sparse triangulations with poor worst-case angle bounds. \n"
    )

    opts = {"kind": "delaunay", "rho2": 1.50}

    vert, etri, tria, tnum = refine(node, edge, [], opts)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"TRIA-MESH: RHO2<=+1.50, |TRIA|={tria.shape[0]}")

    # ---------------------------------------------- RHO2 = +1.00
    print(
        " Setting small values for RHO2, (RHO2 = 1.00 here) generates \n"
        " dense triangulations with good worst-case angle bounds. \n"
    )

    opts = {"kind": "delaunay", "rho2": 1.00}

    vert, etri, tria, tnum = refine(node, edge, [], opts)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"TRIA-MESH: RHO2<=+1.00, |TRIA|={tria.shape[0]}")

    plt.draw()
    plt.show()


def demo2():
    """
    DEMO2 explore impact of refinement "KIND" on mesh quality/density.
    """

    filepath = os.path.dirname(os.path.abspath(__file__))
    meshfile = os.path.join(filepath, "poly_data", "lake.msh")

    node, edge, _, _ = triread(meshfile)

    print(
        " The refine routine supports two Delaunay-based refinement  \n"
        " algorithms: a 'standard' Delaunay-refinement type approach, \n"
        " and a 'Frontal-Delaunay' technique. For problems constrain- \n"
        " ed by element 'quality' alone, the Frontal-Delaunay approa- \n"
        " ch typically produces significantly sparser meshes. In both \n"
        " cases, the same worst-case element quality bounds are sati- \n"
        " sfied in a guaranteed manner. \n"
    )

    # ---------------------------------------------- KIND = "DELAUNAY"
    opts = {"kind": "delaunay", "rho2": 1.00}

    vert, etri, tria, tnum = refine(node, edge, [], opts)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"TRIA-MESH: KIND=DELAUNAY, |TRIA|={tria.shape[0]}")

    # ---------------------------------------------- KIND = "DELFRONT"
    opts = {"kind": "delfront", "rho2": 1.00}

    vert, etri, tria, tnum = refine(node, edge, [], opts)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"TRIA-MESH: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    plt.draw()
    plt.show()


def demo3():
    """
    DEMO3 explore impact of user-defined mesh-size constraints.
    """

    # équivalent de mfilename('fullpath')
    filepath = os.path.dirname(os.path.abspath(__file__))
    meshfile = os.path.join(filepath, "poly_data", "airfoil.msh")

    node, edge, _, _ = triread(meshfile)

    print(
        " Additionally, the refine routine supports size-driven ref- \n"
        " inement, producing meshes that satisfy constraints on elem- \n"
        " ent edge-lengths. The lfshfn routine can be used to create \n"
        " mesh-size functions based on an estimate of the 'local-fea- \n"
        " ture-size' associated with a polygonal domain. The Frontal- \n"
        " Delaunay refinement algorithm discussed in DEMO-2 is espec- \n"
        " ially good at generating high-quality triangulations in the \n"
        " presence of mesh-size constraints. \n"
    )

    # ---------------------------------------------- do size-fun.
    olfs = {"dhdx": 0.15}

    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)

    facecolors = np.mean(hlfs[tlfs[:, 0:3]], axis=1)
    plt.figure()
    plt.tripcolor(
        vlfs[:, 0],
        vlfs[:, 1],
        tlfs[:, 0:3],
        facecolors=facecolors,
        shading="flat",
        cmap="viridis",
    )
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-SIZE: KIND=DELAUNAY, |TRIA|={tlfs.shape[0]}")

    # ---------------------------------------------- do mesh-gen.
    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"TRIA-MESH: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    plt.draw()
    plt.show()


def demo4():
    """
    DEMO4 explore impact of "hill-climbing" mesh optimisations.
    """

    # Équivalent MATLAB mfilename / fileparts
    filepath = os.path.dirname(os.path.abspath(__file__))
    meshfile = os.path.join(filepath, "poly_data", "airfoil.msh")

    node, edge, _, _ = triread(meshfile)

    print(
        " The smooth routine provides iterative mesh 'smoothing' ca- \n"
        " pabilities, seeking to improve triangulation quality by ad- \n"
        " justing the vertex positions and mesh topology. Specifical- \n"
        " ly, a 'hill-climbing' type optimisation is implemented, gu- \n"
        " aranteeing that mesh-quality is improved monotonically. The \n"
        " DRAWSCR routine provides detailed analysis of triangulation \n"
        " quality, plotting histograms of various quality metrics. \n"
    )

    # ---------------------------------------------- do size-fun.
    olfs = {"dhdx": 0.15}

    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)

    # ---------------------------------------------- do mesh-gen.
    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"MESH-REF.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    # ---------------------------------------------- do mesh-opt.
    vnew, enew, tnew, tnum = smooth(vert, etri, tria, tnum)

    plt.figure()
    plt.triplot(vnew[:, 0], vnew[:, 1], tnew[:, 0:3], color=[0.2, 0.2, 0.2])
    plt.gca().set_aspect("equal")
    plt.axis("off")

    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tnew.shape[0]}")

    # ---------------------------------------------- analyse qualité
    hvrt = trihfn(vert, vlfs, tlfs, slfs, hlfs)
    hnew = trihfn(vnew, vlfs, tlfs, slfs, hlfs)

    tricost(vert, etri, tria, tnum, hvrt)
    tricost(vnew, enew, tnew, tnum, hnew)

    plt.show()


def demo5():
    """
    DEMO5 : assemble triangulations for multi-part geometry definitions.
    """

    print(
        "Both refine and smooth routines support multi-part geometry\n"
        "definitions, generating conforming triangulations that respect\n"
        "internal and external constraints.\n"
    )

    # ---------------------------------------------- create geometry

    nod1 = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    edg1 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    tag1 = np.zeros((edg1.shape[0], 1), dtype=int)

    nod2 = np.array([[0.1, 0.0], [0.8, 0.0], [0.8, 0.8], [0.1, 0.8]])

    edg2 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    tag2 = np.ones((edg2.shape[0], 1), dtype=int)

    # Circle geometry
    adel = 2.0 * np.pi / 64.0
    amin = 0.0
    amax = 2.0 * np.pi - adel

    ang = np.arange(amin, amax + adel / 2, adel)
    xcir = 0.33 * np.cos(ang) - 0.33
    ycir = 0.33 * np.sin(ang) - 0.25
    ncir = np.column_stack((xcir, ycir))

    numc = ncir.shape[0]
    ecir = np.column_stack((np.arange(numc - 1), np.arange(1, numc)))
    ecir = np.vstack((ecir, [numc - 1, 0]))
    tagc = np.full((ecir.shape[0], 1), 2, dtype=int)

    # ---------------------------------------------- merge geometries

    # offset inner square indices
    edg2 = edg2 + nod1.shape[0]
    edge = np.vstack((np.hstack((edg1, tag1)), np.hstack((edg2, tag2))))
    node = np.vstack((nod1, nod2))

    # offset circle indices
    ecir = ecir + node.shape[0]
    edge = np.vstack((edge, np.hstack((ecir, tagc))))
    node = np.vstack((node, ncir))

    # ---------------------------------------------- define parts

    edge_tag = edge[:, 2].astype(int)
    part = [
        np.where((edge_tag == 0) | (edge_tag == 1) | (edge_tag == 2))[0],
        np.where(edge_tag == 1)[0],
        np.where(edge_tag == 2)[0],
    ]

    edge = edge[:, :2].astype(int)

    # ---------------------------------------------- size function

    hmax = 0.045
    vlfs, tlfs, hlfs = lfshfn(node, edge, part)
    hlfs = np.minimum(hmax, hlfs)
    slfs = idxtri(vlfs, tlfs)

    # ---------------------------------------------- mesh generation

    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, part, {}, hfun, vlfs, tlfs, slfs, hlfs)

    # ---------------------------------------------- mesh optimization

    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

    # ---------------------------------------------- visualization

    plt.figure()
    for k, color in zip([1, 2, 3], [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]):
        mask = tnum == k
        plt.triplot(vert[:, 0], vert[:, 1], tria[mask, :], color=color, linewidth=0.2)
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    plt.figure()
    tpc = plt.tripcolor(vlfs[:, 0], vlfs[:, 1], tlfs, hlfs, shading="flat")
    plt.colorbar(tpc, label="h")
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"MESH-SIZE: KIND=DELAUNAY, |TRIA|={tlfs.shape[0]}")

    tricost(vert, etri, tria, tnum)

    plt.show()


def demo6():
    """
    DEMO6 build triangulations for geometries with internal constraints.
    """

    print(
        " Both the refine and smooth routines also support geometr- \n"
        " ies containing 'internal' constraints. \n"
    )

    # ---------------------------------------------- create geom.
    node = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [0.2, 0.7],
            [0.6, 0.2],
            [0.4, 0.8],
            [0.0, 0.5],
            [-0.7, 0.3],
            [-0.1, 0.1],
            [-0.6, 0.5],
            [-0.9, -0.8],
            [-0.6, -0.7],
            [-0.3, -0.6],
            [0.0, -0.5],
            [0.3, -0.4],
            [-0.3, 0.4],
            [-0.1, 0.3],
        ]
    )

    edge = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [4, 6],
            [4, 7],
            [4, 8],
            [4, 9],
            [4, 10],
            [4, 11],
            [4, 12],
            [4, 13],
            [4, 14],
            [4, 15],
            [4, 16],
            [4, 17],
            [4, 18],
        ]
    )
    """
    the geometry must be split into its "exterior" and "int-
    erior" components using the optional PART argument. Each
    PART{I} specified should define the "exterior" boundary
    of a polygonal region. "Interior" constraints should not
    be referenced by any polygon in PART -- they are imposed
    as isolated edge constraints.
    """
    part = [np.array([0, 1, 2, 3])]

    # ---------------------------------------------- do size-fun.
    hmax = 0.175

    # ---------------------------------------------- do mesh-gen.
    opts = {"kind": "delaunay"}

    vert, etri, tria, tnum = refine(node, edge, part, opts, hmax)

    # ---------------------------------------------- do mesh-opt.
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, :3], color="k", linewidth=0.2)
    for e in edge:
        plt.plot(node[e, 0], node[e, 1], "k", linewidth=1)

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELAUNAY, |TRIA|={tria.shape[0]}")

    tricost(vert, etri, tria, tnum)

    plt.show()


def demo7():
    """
    DEMO7 investigate the use of quadtree-type mesh refinement.
    """

    filename = __file__
    filepath = "/".join(filename.split("/")[:-1])
    meshfile = f"{filepath}/poly_data/channel.msh"

    node, edge, _, _ = triread(meshfile)

    print(
        " The tridiv routine can also be used to refine existing tr- \n"
        " angulations. Each triangle is split into four new sub-tria- \n"
        " ngles, such that element shape is preserved. Combining the  \n"
        " tridiv and smooth routines allows for hierarchies of high \n"
        " quality triangulations to be generated. \n"
    )

    # ---------------------------------------------- do size-fun.
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)

    pmax = np.max(node, axis=0)
    pmin = np.min(node, axis=0)

    hmax = np.mean(pmax - pmin) / 17.0
    hlfs = np.minimum(hmax, hlfs)

    # ---------------------------------------------- do mesh-gen.
    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)

    # ---------------------------------------------- do mesh-opt.
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    vnew, enew, tnew, tnum = tridiv(vert, etri, tria, tnum)
    vnew, enew, tnew, tnum = smooth(vnew, enew, tnew, tnum)

    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, :3], color="k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    plt.figure()
    plt.triplot(vnew[:, 0], vnew[:, 1], tnew[:, :3], color="k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tnew.shape[0]}")

    tricost(vert, etri, tria, tnum)
    tricost(vnew, enew, tnew, tnum)

    plt.show()


def demo8():
    """
    DEMO8 explore impact of "hill-climbing" mesh optimisations.
    """

    # ---------------------------------------------- create geom.
    node = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [-1.0, 1.0]])
    edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    adel = 2.0 * np.pi / 64.0
    amin = 0.0 * np.pi
    amax = 2.0 * np.pi - adel

    angles = np.arange(amin, amax + adel, adel)
    xcir = 0.20 * np.cos(angles)
    ycir = 0.20 * np.sin(angles)
    ncir = np.column_stack([xcir, ycir])
    numc = ncir.shape[0]

    ecir = np.zeros((numc, 2), dtype=int)
    ecir[:, 0] = np.arange(numc)
    ecir[:, 1] = np.roll(ecir[:, 0], -1)

    ecir = ecir + node.shape[0]
    edge = np.vstack([edge, ecir])
    node = np.vstack([node, ncir])

    # ---------------------------------------------- do mesh-gen.
    hfun = hfun8

    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun)

    # ---------------------------------------------- do mesh-opt.
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

    # ---------------------------------------------- plot mesh
    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, :3], color="k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    # ---------------------------------------------- plot mesh-size function
    plt.figure()
    facecolors = np.mean(hfun8(vert)[tria[:, :3]], axis=1)
    plt.tripcolor(
        vert[:, 0],
        vert[:, 1],
        tria[:, :3],
        facecolors=facecolors,
        edgecolors="none",
        cmap="viridis",
    )
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title("MESH-SIZE function.")

    hvrt = hfun(vert)
    tricost(vert, etri, tria, tnum, hvrt)

    plt.show()


def hfun8(test):
    """
    HFUN8 : user-defined mesh-size function for DEMO-8.

    Parameters
    ----------
    test : ndarray of shape (N,2)
        Coordinates (x,y) at which the mesh-size function is evaluated.

    Returns
    -------
    hfun : ndarray of shape (N,)
        Mesh-size values at input points.
    """
    hmax = 0.05
    hmin = 0.01

    xmid = 0.0
    ymid = 0.0

    hcir = np.exp(-0.5 * (test[:, 0] - xmid) ** 2 - 2.0 * (test[:, 1] - ymid) ** 2)

    hfun = hmax - (hmax - hmin) * hcir

    return hfun


def demo9():
    """
    DEMO9 larger-scale problem, mesh refinement + optimisation.
    """

    # ------------------------------------------- load geometry
    filename = __file__  # current file path
    filepath = "/".join(filename.split("/")[:-1])
    meshfile = f"{filepath}/poly_data/islands.msh"

    # Load input mesh geometry
    node, edge, _, _ = triread(meshfile)

    # ------------------------------------------- do size-function
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)

    # ------------------------------------------- do mesh-generation
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)

    # ------------------------------------------- do mesh-optimisation
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

    # ------------------------------------------- plot triangulated mesh
    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, :3], color="k")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    # ------------------------------------------- plot cost histograms
    tricost(vert, etri, tria, tnum)

    plt.show()


def demo10():
    """
    DEMO10 medium-scale problem mesh refinement + optimisation.
    """

    # ------------------------------------------- load geometry
    filename = __file__
    filepath = "/".join(filename.split("/")[:-1])
    meshfile = f"{filepath}/poly_data/river.msh"

    node, edge, _, _ = triread(meshfile)

    # ------------------------------------------- do size-function
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)

    # ------------------------------------------- do mesh-generation
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)

    # ------------------------------------------- do mesh-optimisation
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

    # ------------------------------------------- plot triangulated mesh
    plt.figure()
    plt.triplot(vert[:, 0], vert[:, 1], tria[:, :3], color="k")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"MESH-OPT.: KIND=DELFRONT, |TRIA|={tria.shape[0]}")

    # ------------------------------------------- plot cost histograms
    tricost(vert, etri, tria, tnum)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m examples.tridemo <demo_number>")
        sys.exit(1)

    try:
        demo_number = int(sys.argv[1])
    except ValueError:
        print("Error: <demo_number> must be an integer.")
        sys.exit(1)

    tridemo(demo_number)
