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


// test Portable::MatrixFree::get_dof_handler()

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <iostream>

#include "../tests.h"


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
  additional_data.mapping_update_flags = update_gradients | update_JxW_values;

  mf_data.reinit(
    MappingQ1<dim>(), dof_handler, constraints, quad, additional_data);

  // Test that get_dof_handler returns the same DoFHandler
  const DoFHandler<dim> &retrieved_dof_handler = mf_data.get_dof_handler();

  // Check that the addresses are the same
  if (&retrieved_dof_handler == &dof_handler)
    {
      deallog << "OK: get_dof_handler() returns the correct DoFHandler"
              << std::endl;
    }
  else
    {
      deallog << "ERROR: get_dof_handler() returns a different DoFHandler"
              << std::endl;
    }

  // Test with explicit index 0
  const DoFHandler<dim> &retrieved_dof_handler_0 = mf_data.get_dof_handler(0);

  if (&retrieved_dof_handler_0 == &dof_handler)
    {
      deallog << "OK: get_dof_handler(0) returns the correct DoFHandler"
              << std::endl;
    }
  else
    {
      deallog << "ERROR: get_dof_handler(0) returns a different DoFHandler"
              << std::endl;
    }

  // Verify we can access properties of the DoFHandler
  deallog << "Number of dofs: " << retrieved_dof_handler.n_dofs() << std::endl;
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
