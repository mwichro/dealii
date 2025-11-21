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

#include "../tests.h"

// Standard headers needed for the "Soft Sanitizer"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>


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

  // We use a unique_ptr to an array of atomics because std::vector<std::atomic>
  // is not copyable/movable and hard to manage in standard containers.
  // 0 = Unlocked, 1 = Locked
  auto cell_locks = std::make_unique<std::atomic<int>[]>(total_cell_slots);

  // Initialize locks to 0
  for (unsigned int i = 0; i < total_cell_slots; ++i)
    cell_locks[i].store(0);

  // Global flag to stop the test immediately if a race occurs
  std::atomic<bool> race_detected(false);

  // Dummy vectors for the interface (not used for calculation)
  LinearAlgebra::distributed::Vector<double> solution, rhs;
  mf_level_storage->initialize_dof_vector(solution);
  mf_level_storage->initialize_dof_vector(rhs);

  // Helper to artificially widen the race window
  auto burn_cycles = []() {
    // Sleep for a tiny amount of time (e.g., 20 microseconds) to allow
    // other threads to potentially collide with us if the coloring is wrong.
    std::this_thread::sleep_for(std::chrono::microseconds(20));
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
          // If another thread failed, stop working
          if (race_detected.load(std::memory_order_relaxed))
            return;

          const auto &patch = patches.get_regular_patch(i);

          // --- LOCK PHASE ---
          // Try to acquire exclusive access to all cells in this patch
          for (const auto &cell_idx : patch.get_cells())
            {
              int expected = 0;
              // atomic::compare_exchange_strong
              // if cell_locks[cell_idx] == 0 (expected), set to 1, return true.
              // if cell_locks[cell_idx] == 1, return false.
              bool success = cell_locks[cell_idx].compare_exchange_strong(
                expected, 1, std::memory_order_acquire);

              if (!success)
                {
                  // RACE CONDITION DETECTED!
                  // This implies another thread is currently processing this
                  // cell.
                  deallog << "RACE DETECTED: Thread collision on cell index "
                          << cell_idx << std::endl;
                  race_detected.store(true);
                  return;
                }
            }

          // --- CRITICAL SECTION SIMULATION ---
          // Hold the locks for a moment to stress test the coloring
          burn_cycles();

          // --- UNLOCK PHASE ---
          // Release access to the cells
          for (const auto &cell_idx : patch.get_cells())
            {
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

  // Ensure TBB is active and has enough threads for a valid test
  // (Assuming global control via environment or deal.II defaults)

  deallog << "Running Atomic Race Detector in 2D..." << std::endl;
  test<2>();

  deallog << "Running Atomic Race Detector in 3D..." << std::endl;
  test<3>();

  deallog << "Tests finished." << std::endl;

  return 0;
}