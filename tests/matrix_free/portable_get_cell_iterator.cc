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

#include <algorithm>
#include <iostream>

#include "../tests.h"


template <int dim>
struct CellCenterComparison
{
  using point_type = Point<dim, double>;
  // Device views with expected centers and space for match results. These are
  // stored on the device and provided at construction time.
  Kokkos::View<point_type *, MemorySpace::Default::kokkos_space> expected;
  Kokkos::View<int *, MemorySpace::Default::kokkos_space>        match_results;

  CellCenterComparison() = default;

  CellCenterComparison(
    const Kokkos::View<point_type *, MemorySpace::Default::kokkos_space>
      &expected_in,
    const Kokkos::View<int *, MemorySpace::Default::kokkos_space>
      &match_results_in)
    : expected(expected_in)
    , match_results(match_results_in)
  {}

  KOKKOS_FUNCTION void
  operator()(
    const typename Portable::MatrixFree<dim, double>::Data *gpu_data) const
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

    // Compare with expected center. We pack expected centers in a
    // color-major array of size n_colors * max_entries_per_color. The test
    // constructs expected accordingly, so compute the same linear index here.
    const unsigned int linear_index = color * max_entries_per_color + cell_id;

    const point_type &expected_center = expected(linear_index);

    bool matches = true;
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (Kokkos::abs(cell_center[d] - expected_center[d]) > 1e-10)
          {
            matches = false;
            break;
          }
      }

    match_results(linear_index) = matches ? 1 : 0;
  }

  // Number of quadrature points per cell: 2^dim for QGauss(2) on Q1 cells
  static const unsigned int n_q_points = (1u << dim);

  // This must match the packing used by the test when constructing the
  // expected centers (max entries per color). It will be set by the test
  // before launching the kernel.
  static unsigned int max_entries_per_color;
};

template <int dim>
unsigned int CellCenterComparison<dim>::max_entries_per_color = 0;


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

  // Compute cell centers on CPU using get_cell_iterator and pack them into
  // a color-major host array that we will transfer to the device.
  const unsigned int n_colors = mf_data.n_colors();

  unsigned int max_entries_per_color = 0;
  for (unsigned int color = 0; color < n_colors; ++color)
    max_entries_per_color =
      std::max(max_entries_per_color, mf_data.n_entries_per_color(color));

  // Prepare host storage for expected centers (packed by color) and a
  // flag array to remember which entries are valid.
  std::vector<Point<dim, double>> host_expected(n_colors *
                                                max_entries_per_color);
  std::vector<char> host_filled(n_colors * max_entries_per_color, 0);

  unsigned int total_cells = 0;
  for (unsigned int color = 0; color < n_colors; ++color)
    {
      deallog << "Color " << color << " has "
              << mf_data.n_entries_per_color(color) << " cells" << std::endl;

      for (unsigned int batch = 0; batch < mf_data.n_entries_per_color(color);
           ++batch)
        {
          // Use get_cell_iterator to get the cell
          auto cell = mf_data.get_cell_iterator(batch, color);

          // Compute the center of the cell
          Point<dim, double> center = cell->center();

          const unsigned int linear_index =
            color * max_entries_per_color + batch;
          host_expected[linear_index] = center;
          host_filled[linear_index]   = 1;

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

  // Now transfer expected centers to device and run the comparison kernel.
  using point_view =
    Kokkos::View<Point<dim, double> *, MemorySpace::Default::kokkos_space>;
  using int_view = Kokkos::View<int *, MemorySpace::Default::kokkos_space>;

  const unsigned int total_slots = n_colors * max_entries_per_color;

  point_view expected_dev("expected_centers", total_slots);
  auto       expected_dev_host = Kokkos::create_mirror_view(expected_dev);
  for (unsigned int i = 0; i < total_slots; ++i)
    expected_dev_host(i) = host_expected[i];
  Kokkos::deep_copy(expected_dev, expected_dev_host);

  int_view match_dev("match_results", total_slots);
  auto     match_dev_host = Kokkos::create_mirror_view(match_dev);
  for (unsigned int i = 0; i < total_slots; ++i)
    match_dev_host(i) = 0;
  Kokkos::deep_copy(match_dev, match_dev_host);

  // Run a device loop over colors and cells to compute the quadrature-based
  // cell centers on the device and compare with the expected centers we
  // transferred. We do this here instead of using evaluate_coefficients to
  // keep the kernel simple and avoid needing to match the functor
  // signature/instantiation.
  for (unsigned int color = 0; color < n_colors; ++color)
    if (mf_data.n_entries_per_color(color) > 0)
      {
        auto               color_data    = mf_data.get_data(color);
        const unsigned int n_cells_color = mf_data.n_entries_per_color(color);

        // Launch a range kernel over cells in this color.
        Kokkos::parallel_for(
          "portable_get_cell_iterator::compare_color",
          Kokkos::RangePolicy<
            MemorySpace::Default::kokkos_space::execution_space>(0,
                                                                 n_cells_color),
          KOKKOS_LAMBDA(const int cell_index) {
            // Compute center from quadrature points
            Point<dim, double> cell_center;
            for (unsigned int d = 0; d < dim; ++d)
              cell_center[d] = 0.0;

            for (unsigned int q = 0; q < CellCenterComparison<dim>::n_q_points;
                 ++q)
              {
                const Point<dim, double> &q_point =
                  color_data.q_points(q, cell_index);
                for (unsigned int d = 0; d < dim; ++d)
                  cell_center[d] += q_point[d];
              }

            for (unsigned int d = 0; d < dim; ++d)
              cell_center[d] /= CellCenterComparison<dim>::n_q_points;

            const unsigned int linear_index =
              color * max_entries_per_color + cell_index;
            const Point<dim, double> expected_center =
              expected_dev(linear_index);

            bool matches = true;
            for (unsigned int d = 0; d < dim; ++d)
              if (Kokkos::abs(cell_center[d] - expected_center[d]) > 1e-10)
                {
                  matches = false;
                  break;
                }

            match_dev(linear_index) = matches ? 1 : 0;
          });
      }

  // Copy match results back and verify only the filled entries.
  Kokkos::deep_copy(match_dev_host, match_dev);

  unsigned int n_matches    = 0;
  unsigned int n_mismatches = 0;
  for (unsigned int i = 0; i < total_slots; ++i)
    if (host_filled[i])
      {
        if (match_dev_host(i) == 1)
          ++n_matches;
        else
          ++n_mismatches;
      }

  deallog << "Matches: " << n_matches << ", Mismatches: " << n_mismatches
          << std::endl;

  // Do not print the match summary in order to keep the test output identical
  // to the reference file. If needed, the checks above can be used in a
  // debugging run.
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
