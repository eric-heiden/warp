// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "hashgrid.h"
#include "sort.h"

extern CUcontext get_current_context();

namespace wp {

template <typename Type>
__global__ void compute_cell_indices(
    HashGrid_t<Type> grid, wp::array_t<vec_t<3, Type>> points, wp::array_t<int> groups, bool use_groups
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < points.shape[0]) {
        const vec_t<3, Type>& point = wp::index(points, tid);
        if (use_groups) {
            const int cell = hash_grid_index(grid, point, wp::index(groups, tid));
            // A negative cell can only occur if group_ids is stale or supplied incorrectly.
            // Keep it as a sentinel; the grouped offset pass ignores it without a host sync.
            grid.point_cells[tid] = cell;
        } else {
            grid.point_cells[tid] = hash_grid_index(grid, point);
        }
        grid.point_ids[tid] = tid;
    }
}

__global__ void compute_cell_offsets(int* cell_starts, int* cell_ends, const int* point_cells, int num_points)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // compute cell start / end
    if (tid < num_points) {
        // scan the particle-cell array to find the start and end
        const int c = point_cells[tid];

        if (tid == 0)
            cell_starts[c] = 0;
        else {
            const int p = point_cells[tid - 1];

            if (c != p) {
                cell_starts[c] = tid;
                cell_ends[p] = tid;
            }
        }

        if (tid == num_points - 1) {
            cell_ends[c] = tid + 1;
        }
    }
}

__global__ void
compute_cell_offsets_checked(int* cell_starts, int* cell_ends, const int* point_cells, int num_points, int num_cells)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points) {
        const int c = point_cells[tid];
        // Stale native group ids produce invalid sentinel cells that must not index the range buffers.
        // CUB can sort those sentinels last, so the first one must close the preceding valid range.
        if (c < 0 || c >= num_cells) {
            const int p = tid > 0 ? point_cells[tid - 1] : -1;
            if (p >= 0 && p < num_cells) {
                cell_ends[p] = tid;
            }
            return;
        }

        // Valid cells are contiguous after sorting. The first point for a cell starts that
        // cell's range; later transitions close the previous valid cell and start this one.
        if (tid == 0)
            cell_starts[c] = 0;
        else {
            const int p = point_cells[tid - 1];

            if (c != p) {
                cell_starts[c] = tid;
                // The previous sorted entry can be an invalid sentinel at the boundary
                // between invalid and valid cells, so only close ranges for valid cells.
                if (p >= 0 && p < num_cells) {
                    cell_ends[p] = tid;
                }
            }
        }

        if (tid == num_points - 1) {
            cell_ends[c] = tid + 1;
        }
    }
}

template <typename Type>
void hash_grid_rebuild_device(
    const wp::HashGrid_t<Type>& grid, const wp::array_t<vec_t<3, Type>>& points, const wp::array_t<int>* groups
)
{
    ContextGuard guard(grid.context);

    int num_points = points.shape[0];
    bool use_groups = groups != nullptr;
    wp::array_t<int> empty_groups;
    const wp::array_t<int>& group_array = groups ? *groups : empty_groups;
    const int num_cells = hash_grid_cell_count(grid);
    if (num_cells < 0) {
        fprintf(stderr, "Warp error: Hash grid cell count overflow in %s\n", __FUNCTION__);
        return;
    }

    wp_launch_device(
        WP_CURRENT_CONTEXT, (wp::compute_cell_indices<Type>), num_points, (grid, points, group_array, use_groups)
    );

    radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_cells, grid.point_ids, num_points);
    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_starts, 0, sizeof(int) * num_cells);
    wp_memset_device(WP_CURRENT_CONTEXT, grid.cell_ends, 0, sizeof(int) * num_cells);

    if (use_groups) {
        // Defensive path for stale native group_ids: invalid cells stay sorted but never index cell ranges.
        wp_launch_device(
            WP_CURRENT_CONTEXT, wp::compute_cell_offsets_checked, num_points,
            (grid.cell_starts, grid.cell_ends, grid.point_cells, num_points, num_cells)
        );
    } else {
        wp_launch_device(
            WP_CURRENT_CONTEXT, wp::compute_cell_offsets, num_points,
            (grid.cell_starts, grid.cell_ends, grid.point_cells, num_points)
        );
    }
}

// Explicit template instantiations
template void
hash_grid_rebuild_device<half>(const HashGrid_t<half>&, const array_t<vec_t<3, half>>&, const array_t<int>*);
template void
hash_grid_rebuild_device<float>(const HashGrid_t<float>&, const array_t<vec_t<3, float>>&, const array_t<int>*);
template void
hash_grid_rebuild_device<double>(const HashGrid_t<double>&, const array_t<vec_t<3, double>>&, const array_t<int>*);


}  // namespace wp
