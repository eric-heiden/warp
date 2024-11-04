# isort: skip_file

from enum import Enum
from typing import Optional

import warp.fem.domain as _domain
import warp.fem.geometry as _geometry
import warp.fem.polynomial as _polynomial

from .function_space import FunctionSpace
from .topology import SpaceTopology
from .basis_space import BasisSpace, PointBasisSpace, ShapeBasisSpace, make_discontinuous_basis_space
from .collocated_function_space import CollocatedFunctionSpace
from .shape import ElementBasis, get_shape_function

from .grid_2d_function_space import make_grid_2d_space_topology

from .grid_3d_function_space import make_grid_3d_space_topology

from .trimesh_function_space import make_trimesh_space_topology

from .tetmesh_function_space import make_tetmesh_space_topology

from .quadmesh_function_space import make_quadmesh_space_topology

from .hexmesh_function_space import make_hexmesh_space_topology

from .nanogrid_function_space import make_nanogrid_space_topology


from .partition import SpacePartition, make_space_partition
from .restriction import SpaceRestriction


from .dof_mapper import DofMapper, IdentityMapper, SymmetricTensorMapper, SkewSymmetricTensorMapper


def make_space_restriction(
    space: Optional[FunctionSpace] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[_domain.GeometryDomain] = None,
    space_topology: Optional[SpaceTopology] = None,
    device=None,
    temporary_store: "Optional[warp.fem.cache.TemporaryStore]" = None,  # noqa: F821
) -> SpaceRestriction:
    """
    Restricts a function space partition to a Domain, i.e. a subset of its elements.

    One of `space_partition`, `space_topology`, or `space` must be provided (and will be considered in that order).

    Args:
        space: (deprecated) if neither `space_partition` nor `space_topology` are provided, the space defining the topology to restrict
        space_partition: the subset of nodes from the space topology to consider
        domain: the domain to restrict the space to, defaults to all cells of the space geometry or partition.
        space_topology: the space topology to be restricted, if `space_partition` is ``None``.
        device: device on which to perform and store computations
        temporary_store: shared pool from which to allocate temporary arrays
    """

    if space_partition is None:
        if space_topology is None:
            assert space is not None
            space_topology = space.topology

        if domain is None:
            domain = _domain.Cells(geometry=space_topology.geometry)

        space_partition = make_space_partition(
            space_topology=space_topology, geometry_partition=domain.geometry_partition
        )
    elif domain is None:
        domain = _domain.Cells(geometry=space_partition.geo_partition)

    return SpaceRestriction(
        space_partition=space_partition, domain=domain, device=device, temporary_store=temporary_store
    )


def make_polynomial_basis_space(
    geo: _geometry.Geometry,
    degree: int = 1,
    element_basis: Optional[ElementBasis] = None,
    discontinuous: bool = False,
    family: Optional[_polynomial.Polynomial] = None,
) -> BasisSpace:
    """
    Equips a geometry with a polynomial basis.

    Args:
        geo: the Geometry on which to build the space
        degree: polynomial degree of the per-element shape functions
        discontinuous: if True, use Discontinuous Galerkin shape functions. Discontinuous is implied if degree is 0, i.e, piecewise-constant shape functions.
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the constructed basis space
    """

    base_geo = geo.base if isinstance(geo, _geometry.DeformedGeometry) else geo

    if element_basis is None:
        element_basis = ElementBasis.LAGRANGE
    elif element_basis == ElementBasis.SERENDIPITY and degree == 1:
        # Degree-1 serendipity is always equivalent to Lagrange
        element_basis = ElementBasis.LAGRANGE

    shape = get_shape_function(geo.reference_cell(), geo.dimension, degree, element_basis, family)

    if discontinuous or degree == 0 or element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
        return make_discontinuous_basis_space(geo, shape)

    topology = None
    if isinstance(base_geo, _geometry.Grid2D):
        topology = make_grid_2d_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Grid3D):
        topology = make_grid_3d_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Trimesh):
        topology = make_trimesh_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Tetmesh):
        topology = make_tetmesh_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Quadmesh):
        topology = make_quadmesh_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Hexmesh):
        topology = make_hexmesh_space_topology(geo, shape)
    elif isinstance(base_geo, _geometry.Nanogrid) or isinstance(base_geo, _geometry.AdaptiveNanogrid):
        topology = make_nanogrid_space_topology(geo, shape)

    if topology is None:
        raise NotImplementedError(f"Unsupported geometry type {geo.name}")

    return ShapeBasisSpace(topology, shape)


def make_collocated_function_space(
    basis_space: BasisSpace, dtype: type = float, dof_mapper: Optional[DofMapper] = None
) -> CollocatedFunctionSpace:
    """
    Constructs a function space from a basis space and a value type, such that all degrees of freedom of the value type are stored at each of the basis nodes.

    Args:
        geo: the Geometry on which to build the space
        dtype: value type the function space. If ``dof_mapper`` is provided, the value type from the DofMapper will be used instead.
        dof_mapper: mapping from node degrees of freedom to function values, defaults to Identity. Useful for reduced coordinates, e.g. :py:class:`SymmetricTensorMapper` maps 2x2 (resp 3x3) symmetric tensors to 3 (resp 6) degrees of freedom.

    Returns:
        the constructed function space
    """
    return CollocatedFunctionSpace(basis_space, dtype=dtype, dof_mapper=dof_mapper)


def make_polynomial_space(
    geo: _geometry.Geometry,
    dtype: type = float,
    dof_mapper: Optional[DofMapper] = None,
    degree: int = 1,
    element_basis: Optional[ElementBasis] = None,
    discontinuous: bool = False,
    family: Optional[_polynomial.Polynomial] = None,
) -> CollocatedFunctionSpace:
    """
    Equips a geometry with a collocated, polynomial function space.
    Equivalent to successive calls to :func:`make_polynomial_basis_space` and `make_collocated_function_space`.

    Args:
        geo: the Geometry on which to build the space
        dtype: value type the function space. If ``dof_mapper`` is provided, the value type from the DofMapper will be used instead.
        dof_mapper: mapping from node degrees of freedom to function values, defaults to Identity. Useful for reduced coordinates, e.g. :py:class:`SymmetricTensorMapper` maps 2x2 (resp 3x3) symmetric tensors to 3 (resp 6) degrees of freedom.
        degree: polynomial degree of the per-element shape functions
        discontinuous: if True, use Discontinuous Galerkin shape functions. Discontinuous is implied if degree is 0, i.e, piecewise-constant shape functions.
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the constructed function space
    """

    basis_space = make_polynomial_basis_space(geo, degree, element_basis, discontinuous, family)
    return CollocatedFunctionSpace(basis_space, dtype=dtype, dof_mapper=dof_mapper)
