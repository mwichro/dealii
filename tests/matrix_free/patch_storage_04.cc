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


// It creates a PatchStorage object with threads and verifies that
// no race conditions occur using an atomic-lock "Soft Sanitizer" approach.


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>

#include "deal.II/dofs/dof_handler.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/patch_storage.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "../tests.h"


template <int dim>
void
test()
{
  // 1. Generate triangulation
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);

  const unsigned int test_level = 4;

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(test_level);

  // 2. Initialize FE, DoFHandler, Mapping
  const unsigned int fe_degree = 1;
  FE_DGQ<dim>        fe(fe_degree);
  DoFHandler<dim>    dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  MappingQ1<dim> mapping;

  // 3. Initialize constraints
  AffineConstraints<double> constraints;
  IndexSet                  locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  constraints.reinit(locally_relevant_dofs);
  constraints.close();

  // 4. Initialize MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mg_level          = test_level;
  additional_data.store_ghost_cells = true;

  std::shared_ptr<MatrixFree<dim, double>> mf_level_storage =
    std::make_shared<MatrixFree<dim, double>>();
  mf_level_storage->reinit(mapping,
                           dof_handler,
                           constraints,
                           QGauss<1>(fe_degree + 1),
                           additional_data);

  // 5. Build PatchStorage with threads
  typename PatchStorage<MatrixFree<dim, double>>::AdditionalData
    with_threads_data;
  with_threads_data.tasks_parallel_scheme =
    PatchStorage<MatrixFree<dim, double>>::AdditionalData::by_color;

  PatchStorage<MatrixFree<dim, double>> patch_storage_with_threads(
    mf_level_storage);
  patch_storage_with_threads.initialize(with_threads_data);

  // ========================================================================
  // 6. "Soft Thread Sanitizer" Setup
  // ========================================================================

  // We need to verify that multiple threads do not access the same cell
  // simultaneously. In MatrixFree, cells are stored in batches.
  // We calculate the total number of cell slots (VectorizedArray size *
  // batches).
  const unsigned int n_lanes          = VectorizedArray<double>::size();
  const unsigned int n_cell_batches   = mf_level_storage->n_cell_batches();
  const unsigned int total_cell_slots = n_cell_batches * n_lanes;

  // Use a unique_ptr to avoid large stack allocations or std::vector issues
  // with atomics
  auto cell_locks = std::make_unique<std::atomic<int>[]>(total_cell_slots);

  // Initialize locks to 0 (Unlocked)
  for (unsigned int i = 0; i < total_cell_slots; ++i)
    cell_locks[i].store(0);

  // Global flag to stop the test immediately if a race occurs
  std::atomic<bool> race_detected(false);

  // Dummy vectors
  LinearAlgebra::distributed::Vector<double> solution, rhs;
  mf_level_storage->initialize_dof_vector(solution);
  mf_level_storage->initialize_dof_vector(rhs);

  // STRICT HELPER: Busy Wait
  // We do NOT use sleep_for. Sleep yields the CPU.
  // We want to hold the CPU core to force physical overlap with other threads.
  auto burn_cycles = []() {
    volatile int x = 0;
    // 5000 iterations is enough to create a window for collision
    // without freezing the test for too long.
    for (int i = 0; i < 5000; ++i)
      x++;
  };

  // 7. Define the Race Detector Worker
  const auto patch_worker =
    [&](const PatchStorage<MatrixFree<dim, double>> &patches,
        LinearAlgebra::distributed::Vector<double> & /*dst*/,
        const LinearAlgebra::distributed::Vector<double> & /*src*/,
        const typename PatchStorage<MatrixFree<dim, double>>::PatchRange &range,
        const unsigned int /*thread_id*/) {
      for (unsigned int i = range.first; i < range.second; ++i)
        {
          if (race_detected.load(std::memory_order_relaxed))
            return;

          const auto &patch = patches.get_regular_patch(i);
          const auto &cells = patch.get_cells();

          // --- LOCK PHASE ---
          for (const auto &cell_idx : cells)
            {
              // Safety check to ensure we don't segfault on invalid indices
              Assert(cell_idx < total_cell_slots,
                     ExcMessage("Cell index out of bounds for atomic array."));

              int expected = 0;
              // Try to switch 0 -> 1.
              // memory_order_acquire ensures we see up-to-date values from
              // other threads
              bool success = cell_locks[cell_idx].compare_exchange_strong(
                expected, 1, std::memory_order_acquire);

              if (!success)
                {
                  deallog << "RACE DETECTED: Collision on cell index "
                          << cell_idx << " in patch " << i << std::endl;
                  race_detected.store(true);
                  return;
                }
            }

          // --- STRICT CRITICAL SECTION ---
          // Keep the lock held while burning CPU cycles.
          burn_cycles();

          // --- UNLOCK PHASE ---
          for (const auto &cell_idx : cells)
            {
              // memory_order_release ensures our lock release is visible to
              // others
              cell_locks[cell_idx].store(0, std::memory_order_release);
            }
        }
    };

  // 8. Execute the patch loop
  // Ensure we actually have threads, otherwise the test is trivial.
  if (MultithreadInfo::n_threads() > 1)
    deallog << "Running race detection with multiple threads." << std::endl;
  else
    deallog << "WARNING: Running in serial. Race detection is trivial."
            << std::endl;

  patch_storage_with_threads.patch_loop(patch_worker, solution, rhs);

  // 9. Verification
  if (race_detected.load())
    {
      AssertThrow(false,
                  ExcMessage(
                    "Parallel race condition detected! "
                    "Two patches in the same color accessed the same cell."));
    }
  else
    {
      deallog << "No race conditions detected." << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

  initlog();

  deallog << "Running Atomic Race Detector in 2D..." << std::endl;
  test<2>();

  deallog << "Running Atomic Race Detector in 3D..." << std::endl;
  test<3>();

  deallog << "Tests finished." << std::endl;

  return 0;
}