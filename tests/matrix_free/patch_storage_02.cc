// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2020 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

// This test verifies that PatchStorage correctly constructs patches.
// Check if for each coarse cell on test_level-1, there exists exactly one patch
// on test_level that contains all its children in the correct (lexicographical)
// order.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/patch_storage.h>

#include "../tests.h"


template <int dim>
void
test()
{
  // 1. Generate triangulation
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);

  const unsigned int test_level = 3;

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(test_level); // Refine once

  // 2. Initialize FE, DoFHandler, Mapping
  const unsigned int fe_degree = 1;
  FE_DGQ<dim>        fe(fe_degree);
  DoFHandler<dim>    dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  const unsigned int n_lanes = VectorizedArray<double>::size();

  MappingQ1<dim> mapping; // Using Q1 mapping as in step-94

  // 3. Initialize constraints (needed for MatrixFree)
  AffineConstraints<double> constraints;
  IndexSet                  locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  constraints.reinit(locally_relevant_dofs);
  // No boundary conditions or hanging nodes needed for this simple test yet
  constraints.close();

  // 4. Initialize MatrixFree for test_level
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mg_level          = test_level; // Target level 1
  additional_data.store_ghost_cells = true;

  std::shared_ptr<MatrixFree<dim, double>> mf_level_storage =
    std::make_shared<MatrixFree<dim, double>>();
  mf_level_storage->reinit(mapping,
                           dof_handler,
                           constraints,
                           QGauss<1>(fe_degree + 1),
                           additional_data);

  // 5. Build PatchStorage on test_level
  PatchStorage<MatrixFree<dim, double>> patch_storage(mf_level_storage);
  patch_storage.initialize();

  // 6. Check that all patches are constructed correctly.

  for (const auto &coarse_cell :
       triangulation.cell_iterators_on_level(test_level - 1))
    {
      // Collect all descendant cells on the fine level (level 3) in
      // lexicographical order via recursive traversal.
      std::vector<typename Triangulation<dim>::level_cell_iterator>
        descendant_cells;
      descendant_cells.resize(coarse_cell->n_children());

      for (unsigned int c = 0; c < coarse_cell->n_children(); ++c)
        descendant_cells[c] = coarse_cell->child(c);


      const unsigned int n_patches              = patch_storage.n_patches();
      unsigned int       n_found_matching_patch = 0;

      for (unsigned int patch_idx = 0; patch_idx < n_patches; ++patch_idx)
        {
          const auto &patch = patch_storage.get_regular_patch(patch_idx);
          const auto  patch_cells_ids = patch.get_cells();
          std::vector<typename Triangulation<dim>::level_cell_iterator>
            patch_cells(descendant_cells.size());
          for (unsigned int i = 0; i < patch_cells.size(); ++i)
            {
              const auto &cell_index = patch_cells_ids[i];
              patch_cells[i] =
                mf_level_storage->get_cell_iterator(cell_index / n_lanes,
                                                    cell_index % n_lanes);
            }

          if (patch_cells_ids.size() != descendant_cells.size())
            continue;

          bool match = true;
          for (unsigned int i = 0; i < patch_cells.size(); ++i)
            {
              if (patch_cells[i] != descendant_cells[i])
                {
                  match = false;
                  break;
                }
            }

          if (match)
            {
              n_found_matching_patch++;
            }
        }

      AssertThrow(n_found_matching_patch == 1,
                  ExcMessage(
                    "No matching patch found for coarse cell at level " +
                    std::to_string(coarse_cell->level()) + ", index " +
                    std::to_string(coarse_cell->index())));
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  initlog();

  deallog << "Running test in 2D..." << std::endl;
  test<2>();

  deallog << "Running test in 3D..." << std::endl;
  test<3>();

  deallog << "Tests finished." << std::endl;

  return 0; // Indicate success
}
