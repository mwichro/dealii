// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// test Portable::MatrixFree::get_cell_iterator()

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <iostream>

#include "../tests.h"


template <int dim>
struct CellCenterComparison
{
  using point_type = Point<dim, double>;

  KOKKOS_FUNCTION void
  operator()(
    const typename Portable::MatrixFree<dim, double>::Data *gpu_data,
    const Kokkos::View<point_type *, MemorySpace::Default::kokkos_space>
                        expected_centers,
    Kokkos::View<int *, MemorySpace::Default::kokkos_space> match_results) const
  {
    const unsigned int cell_id = gpu_data->cell_index;
    const unsigned int color   = gpu_data->precomputed_data->row_start /
                               gpu_data->precomputed_data->padding_length;

    // Get the cell center from the quadrature points (approximate)
    point_type cell_center;
    for (unsigned int d = 0; d < dim; ++d)
      cell_center[d] = 0.0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const point_type &q_point =
          gpu_data->precomputed_data->q_points(q, cell_id);
        for (unsigned int d = 0; d < dim; ++d)
          cell_center[d] += q_point[d];
      }

    for (unsigned int d = 0; d < dim; ++d)
      cell_center[d] /= n_q_points;

    // Compare with expected center
    const point_type &expected = expected_centers(color * 1000 + cell_id);

    bool matches = true;
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (Kokkos::abs(cell_center[d] - expected[d]) > 1e-10)
          {
            matches = false;
            break;
          }
      }

    match_results(color * 1000 + cell_id) = matches ? 1 : 0;
  }

  static const unsigned int n_q_points = 8; // 2^dim for Q1 elements with Q2
};


template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.close();

  const QGauss<1> quad(2);

  Portable::MatrixFree<dim, double> mf_data;

  typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;

  mf_data.reinit(
    MappingQ1<dim>(), dof_handler, constraints, quad, additional_data);

  // Get the colored graph to iterate over cells
  const auto &graph = mf_data.get_colored_graph();

  deallog << "Number of colors: " << graph.size() << std::endl;

  // Compute cell centers on CPU using get_cell_iterator
  std::vector<Point<dim, double>> expected_centers;
  unsigned int                    total_cells = 0;

  for (unsigned int color = 0; color < graph.size(); ++color)
    {
      deallog << "Color " << color << " has " << graph[color].size()
              << " cells" << std::endl;

      for (unsigned int batch = 0; batch < graph[color].size(); ++batch)
        {
          // Use get_cell_iterator to get the cell
          auto cell = mf_data.get_cell_iterator(batch, color);

          // Compute the center of the cell
          Point<dim, double> center = cell->center();

          expected_centers.push_back(center);

          deallog << "  Cell (batch=" << batch << ", color=" << color
                  << "): center = " << center << ", level = " << cell->level()
                  << ", index = " << cell->index() << std::endl;

          total_cells++;
        }
    }

  deallog << "Total cells processed: " << total_cells << std::endl;

  // Verify consistency with the triangulation
  unsigned int n_active_cells = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        n_active_cells++;
    }

  deallog << "Active cells in triangulation: " << n_active_cells << std::endl;

  if (total_cells == n_active_cells)
    {
      deallog << "OK: Number of cells matches" << std::endl;
    }
  else
    {
      deallog << "ERROR: Number of cells mismatch" << std::endl;
    }
}


int
main()
{
  initlog();

  Kokkos::initialize();
  {
    test<2>();
    test<3>();
  }
  Kokkos::finalize();

  return 0;
}
