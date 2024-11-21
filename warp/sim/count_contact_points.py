# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Collision handling functions and kernels.
"""

import numpy as np

import warp as wp
from warp.sim.model import Model

from .model import PARTICLE_FLAG_ACTIVE, ModelShapeGeometry

@wp.kernel(enable_backward=False)
def count_contact_points(
    contact_pairs: wp.array(dtype=int, ndim=2),
    geo: ModelShapeGeometry,
    mesh_contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    if shape_b == -1:
        actual_shape_a = shape_a
        actual_type_a = geo.type[shape_a]
        # ground plane
        actual_type_b = wp.sim.GEO_PLANE
        actual_shape_b = -1
    else:
        type_a = geo.type[shape_a]
        type_b = geo.type[shape_b]
        # unique ordering of shape pairs
        if type_a < type_b:
            actual_shape_a = shape_a
            actual_shape_b = shape_b
            actual_type_a = type_a
            actual_type_b = type_b
        else:
            actual_shape_a = shape_b
            actual_shape_b = shape_a
            actual_type_a = type_b
            actual_type_b = type_a

    # determine how many contact points need to be evaluated
    num_contacts = 0
    num_actual_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
        num_actual_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        if actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 2  # vertex-based collision for infinite plane
                num_actual_contacts = 2
            else:
                num_contacts = 2 + 4  # vertex-based collision + plane edges
                num_actual_contacts = 2 + 4
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
            num_actual_contacts = num_contacts_a + num_contacts_b
        else:
            num_contacts = 2
            num_actual_contacts = 2
    elif actual_type_a == wp.sim.GEO_BOX:
        if actual_type_b == wp.sim.GEO_BOX:
            num_contacts = 24
            num_actual_contacts = 24
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
            num_actual_contacts = num_contacts_a + num_contacts_b
        elif actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 8  # vertex-based collision
                num_actual_contacts = 8
            else:
                num_contacts = 8 + 4  # vertex-based collision + plane edges
                num_actual_contacts = 8 + 4
        else:
            num_contacts = 8
    elif actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo.source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        if mesh_contact_max > 0:
            num_contacts_a = wp.min(mesh_contact_max, num_contacts_a)
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
        else:
            num_contacts_b = 0
        num_contacts = num_contacts_a + num_contacts_b
        num_actual_contacts = num_contacts_a + num_contacts_b
    elif actual_type_a == wp.sim.GEO_PLANE:
        return  # no plane-plane contacts
    else:
        wp.printf(
            "count_contact_points: unsupported geometry type combination %d and %d\n", actual_type_a, actual_type_b
        )

    wp.atomic_add(contact_count, 0, num_contacts)
    wp.atomic_add(contact_count, 1, num_actual_contacts)