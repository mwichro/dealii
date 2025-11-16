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


// It creates two PatchStorage objects with  and without threads and  verifies
// that they contain the exact same set of patches.


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

  // 5. Build PatchStorage on test_level without threads
  typename PatchStorage<MatrixFree<dim, double>>::AdditionalData
    no_threads_data;
  no_threads_data.tasks_parallel_scheme =
    PatchStorage<MatrixFree<dim, double>>::AdditionalData::none;

  PatchStorage<MatrixFree<dim, double>> patch_storage_reference(
    mf_level_storage);
  patch_storage_reference.initialize(no_threads_data);

  // 6. Build PatchStorage on test_level with threads
  typename PatchStorage<MatrixFree<dim, double>>::AdditionalData
    with_threads_data;
  with_threads_data.tasks_parallel_scheme =
    PatchStorage<MatrixFree<dim, double>>::AdditionalData::none;

  PatchStorage<MatrixFree<dim, double>> patch_storage_with_threads(
    mf_level_storage);
  patch_storage_with_threads.initialize(with_threads_data);

  // 7. Check that both PatchStorage store identical patches
  Assert(patch_storage_reference.n_patches() ==
           patch_storage_with_threads.n_patches(),
         ExcInternalError());

  for (unsigned int patch_idx = 0;
       patch_idx < patch_storage_reference.n_patches();
       ++patch_idx)
    {
      const auto &patch_info_ref =
        patch_storage_reference.get_regular_patch(patch_idx);

      unsigned int n_found_match = false;
      for (unsigned int other_patch_idx = 0;
           other_patch_idx < patch_storage_with_threads.n_patches();
           ++other_patch_idx)
        {
          const auto &patch_info_other =
            patch_storage_with_threads.get_regular_patch(other_patch_idx);
          if (patch_info_ref.get_cells() == patch_info_other.get_cells())
            {
              n_found_match++;
            }
        }
      AssertThrow(n_found_match == 1,
                  ExcMessage("Could not find a matching patch in the threaded "
                             "PatchStorage for patch " +
                             std::to_string(patch_idx) + " from reference."));
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
