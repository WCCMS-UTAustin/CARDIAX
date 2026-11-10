import numpy as onp


class NodeMask:
    """Explicit selection of mesh nodes, usable wherever a location function is.

    A location function is evaluated on coordinates, so it cannot follow a
    boundary once the mesh has moved (e.g. a planar ``top`` after a load step).
    A NodeMask records *which nodes* make up a boundary, independent of where
    they currently are. Tag the boundary once on a reference mesh, then reuse the
    mask for any FE object built on a mesh with the same node / cell ordering.

    A face is selected when all of its vertices are tagged; Dirichlet nodes are
    the tagged nodes themselves.

    Parameters
    ----------
    mask : array_like of bool, shape (num_nodes,)
        True for the nodes on the boundary.

    Examples
    --------
    >>> top_mask = fe.tag_nodes(top)            # on the reference mesh
    >>> fe_new = FiniteElement(deformed_mesh, ...)
    >>> problem = Problem({"u": fe_new}, location_fns={"u": {"top": top_mask}})
    """

    def __init__(self, mask):
        mask = onp.asarray(mask)
        if mask.dtype != bool:
            raise TypeError(f"NodeMask requires a boolean array, got dtype {mask.dtype}.")
        if mask.ndim != 1:
            raise ValueError(f"NodeMask requires a 1D array, got shape {mask.shape}.")
        self.mask = mask

    @classmethod
    def from_location_fn(cls, fe, location_fn):
        """Tag the nodes of ``fe`` where ``location_fn`` is True (coordinate-based)."""
        import jax
        return cls(onp.array(jax.vmap(location_fn)(fe.nodes)).astype(bool).reshape(-1))

    def check_compatible(self, num_nodes):
        if self.mask.shape[0] != num_nodes:
            raise ValueError(f"NodeMask has {self.mask.shape[0]} entries but the mesh has "
                             f"{num_nodes} nodes. A mask can only be reused on a mesh with "
                             "the same node ordering it was tagged on.")

    @property
    def node_inds(self):
        return onp.argwhere(self.mask).reshape(-1)

    def __len__(self):
        return self.mask.shape[0]

    def __repr__(self):
        return f"NodeMask({int(self.mask.sum())} of {self.mask.shape[0]} nodes tagged)"
