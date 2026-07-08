import numpy as np


def loadmsh(name):
    """
    Load a `.MSH` file for JIGSAW.

    This function reads a JIGSAW-formatted mesh file (`.MSH`) and returns
    a structured dictionary representing the mesh data. The file may contain
    various mesh types, including Euclidean meshes, ellipsoidal meshes,
    and structured grids.

    Parameters
    ----------
    name : str
        Path or base name of the `.MSH` file to load (without extension).

    Returns
    -------
    mesh : dict
        Dictionary containing the mesh entities and metadata. The structure
        depends on the mesh type (`mesh["MSHID"]`):

        **If `mesh["MSHID"] == "EUCLIDEAN-MESH"`:**
            - `POINT.COORD` : (NP, ND+1) ndarray
              Point coordinates and IDs.
            - `POINT.POWER` : (NP,) ndarray
              Vertex weights for the dual “power” tessellation.
            - `EDGE2.INDEX`, `TRIA3.INDEX`, `QUAD4.INDEX`, `TRIA4.INDEX`,
              `HEXA8.INDEX`, `WEDG6.INDEX`, `PYRA5.INDEX` : ndarrays
              Connectivity and IDs for 1D–3D mesh elements.
            - `BOUND.INDEX` : (NB, 3) ndarray
              Boundary-to-part associations and element type tags.
            - `VALUE` : (NP, NV) ndarray
              Scalar or vector values at mesh vertices.
            - `SLOPE` : (NP,) ndarray
              Gradient-limit values `||dh/dx||` used by the Eikonal solver `MARCHE`.

        **If `mesh["MSHID"] == "ELLIPSOID-MESH"`:**
            - `RADII` : (3,) ndarray
              Principal ellipsoid radii.
            - Additional fields as in `"EUCLIDEAN-MESH"` may also be included.

        **If `mesh["MSHID"] == "EUCLIDEAN-GRID"` or `"ELLIPSOID-GRID"`:**
            - `POINT.COORD` : list of ND arrays
              Grid coordinates along each spatial axis.
            - `VALUE` : (NM, NV) ndarray
              Values at each grid vertex (`NM` = product of grid dimensions).
            - `SLOPE` : (NM,) ndarray
              Gradient-limits for each grid vertex.

    Notes
    -----
    - This function automatically detects the mesh type from the file header.
    - Optional entities are only loaded if present in the `.MSH` file.
    - `BOUND.INDEX` defines part boundaries and topology using internal
      constants from the JIGSAW `LIBDATA` specification.

    See Also
    --------
    jigsaw : Run JIGSAW mesh generator.
    savemsh : Save mesh data to a `.MSH` file.

    References
    ----------
    Translation of the MESH2D function `loadmsh`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    mesh = {}

    try:
        with open(name, "r") as ffid:
            _real = float
            kind = "EUCLIDEAN-MESH"
            nver = 0
            ndim = 0

            while True:
                # -- read next line from file
                lstr = ffid.readline()
                if not lstr:
                    break

                lstr = lstr.strip()
                if len(lstr) == 0 or lstr[0] == "#":
                    continue

                # -- tokenise line about '=' character
                tstr = lstr.lower().split("=")
                if len(tstr) != 2:
                    print(f"Warning: Invalid tag: {lstr}")
                    continue

                key, value = tstr[0].strip(), tstr[1].strip()

                if key == "mshid":
                    # -- read "MSHID" data
                    stag = value.split(";")
                    nver = int(stag[0])
                    if len(stag) >= 2:
                        kind = stag[1].strip().upper()

                elif key == "ndims":
                    # -- read "NDIMS" data
                    ndim = int(value)

                elif key == "radii":
                    # -- read "RADII" data
                    stag = value.split(";")
                    if len(stag) == 3:
                        mesh["radii"] = [float(stag[0]), float(stag[1]), float(stag[2])]
                    else:
                        print(f"Warning: Invalid RADII: {lstr}")

                elif key == "point":
                    # -- read "POINT" data
                    nnum = int(value)
                    data = np.loadtxt(ffid, max_rows=nnum, delimiter=";")
                    mesh["point"] = {"coord": data}

                elif key == "coord":
                    # -- read "COORD" data
                    stag = value.split(";")
                    idim = int(stag[0])
                    cnum = int(stag[1])
                    data = np.loadtxt(ffid, max_rows=cnum, delimiter=";")
                    if "point" not in mesh:
                        mesh["point"] = {}
                    if "coord" not in mesh["point"]:
                        mesh["point"]["coord"] = {}
                    mesh["point"]["coord"][idim] = data

                elif key in [
                    "edge2",
                    "tria3",
                    "quad4",
                    "tria4",
                    "hexa8",
                    "wedg6",
                    "pyra5",
                    "bound",
                ]:
                    # Read element data
                    nnum = int(value)
                    data = np.loadtxt(ffid, max_rows=nnum, delimiter=";")
                    mesh[key] = {"index": data}

                elif key in ["value", "slope", "power"]:
                    # Read "VALUE", "SLOPE", or "POWER" data
                    stag = value.split(";")
                    nnum = int(stag[0])
                    vnum = int(stag[1])
                    numr = nnum * vnum
                    data = np.fromfile(ffid, count=numr, sep=";").reshape(nnum, vnum)

                    mesh[key] = data

            mesh["mshID"] = kind
            mesh["fileV"] = nver

            # Reshape grid data if necessary
            if ndim > 0 and kind in ["EUCLIDEAN-GRID", "ELLIPSOID-GRID"]:
                if "value" in mesh and "point" in mesh:
                    if ndim == 2:
                        mesh["value"] = mesh["value"].reshape(
                            len(mesh["point"]["coord"][1]),
                            len(mesh["point"]["coord"][0]),
                            -1,
                        )
                    elif ndim == 3:
                        mesh["value"] = mesh["value"].reshape(
                            len(mesh["point"]["coord"][1]),
                            len(mesh["point"]["coord"][0]),
                            len(mesh["point"]["coord"][2]),
                            -1,
                        )
                if "slope" in mesh and "point" in mesh:
                    if ndim == 2:
                        mesh["slope"] = mesh["slope"].reshape(
                            len(mesh["point"]["coord"][1]),
                            len(mesh["point"]["coord"][0]),
                            -1,
                        )
                    elif ndim == 3:
                        mesh["slope"] = mesh["slope"].reshape(
                            len(mesh["point"]["coord"][1]),
                            len(mesh["point"]["coord"][0]),
                            len(mesh["point"]["coord"][2]),
                            -1,
                        )

    except Exception as err:
        print(f"Error reading file {name}: {err}")
        raise

    return mesh
