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


#ifndef dealii__patch_storage_h
#define dealii__patch_storage_h



#include <deal.II/base/data_out_base.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>



DEAL_II_NAMESPACE_OPEN

#ifndef DOXYGEN

namespace internal
{
  namespace GaussSeidel
  {
    class TaskInfoDummy
    {
    public:
      using cell_index_type_scalar = unsigned int;
      const static constexpr unsigned int invalid_category =
        std::numeric_limits<unsigned int>::max();
      using cell_index_type_vector = std::array<unsigned int, 1>;
      const static constexpr unsigned int n_categories = 1;

      TaskInfoDummy()
        : current_communicate_index(0)
      {}
      unsigned int current_communicate_index;

      template <typename number>
      void
      communicate(const unsigned int,
                  const int,
                  ::dealii::LinearAlgebra::distributed::Vector<number> &) const
      {}

      template <typename... Args>
      void
      initialize(const Args &...args)
      {
        (void)sizeof...(args);
      }


      unsigned int
      determine_minimum_category_of_patch(
        const std::set<types::global_dof_index> &,
        const std::vector<unsigned int> &) const
      {
        return 0;
      }

      static unsigned int
      determine_cell_partition(const unsigned int &)
      {
        return 0;
      }
    };

    unsigned int
    min_category(const unsigned int a, const unsigned int b);

  } // namespace GaussSeidel
} // namespace internal


#endif // DOXYGEN

/**
 * Base class for vertex patches.
 *
 * Provides common interface and functionality for both regular and general
 * vertex patches, including methods to check for overlap and determine if
 * patches contain ghost cells.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class VertexPatchBase
{
public:
  /**
   * Type used to represent a unique index for each cell.
   */
  using CellIndex = unsigned int;

  /**
   * Alias for the cell iterator type of the underlying Triangulation.
   */
  using CellIterator = typename Triangulation<dim>::cell_iterator;

  /**
   * Virtual destructor.
   */
  virtual ~VertexPatchBase() = default;

  /**
   * Pure virtual function that returns the cells in this patch as a vector.
   * Must be implemented by derived classes.
   *
   * @return A vector of CellIndex values representing the cells in the patch.
   */
  virtual std::vector<CellIndex>
  get_cells_vector() const = 0;

  /**
   * Check if this patch has an overlap with another patch.
   *
   * Two patches overlap if they share any cell indices. This is used
   * for graph coloring to ensure that patches that overlap cannot be
   * processed in parallel.
   *
   * @param other_patch The other patch to check against.
   * @return true if the patches share at least one cell, false otherwise.
   */
  bool
  has_overlap_with(const VertexPatchBase &other_patch) const
  {
    const std::vector<CellIndex> this_cells  = get_cells_vector();
    const std::vector<CellIndex> other_cells = other_patch.get_cells_vector();

    // Check if any cell indices are shared
    for (const auto &this_cell : this_cells)
      for (const auto &other_cell : other_cells)
        if (this_cell == other_cell)
          return true;

    return false;
  }

  /**
   * Check if the patch is partially ghosted.
   *
   * A patch is partially ghosted if at least one of its cells is a ghost
   * cell (not locally owned on the current MPI process).
   *
   * @return true if the patch contains at least one ghost cell, false
   * otherwise.
   */
  bool
  is_partially_ghosted() const
  {
    return partially_ghosted;
  }

protected:
  /**
   * Flag indicating whether this patch contains at least one ghost cell.
   * Set by derived classes during construction.
   */
  bool partially_ghosted;
};

/**
 * Represents a regular patch, i.e., a patch centered at a vertex
 * with exactly 2^dim cells.
 *
 * Stores the cell indices and potentially orientation information. Provides
 * methods to check constructibility and access cell data.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
struct RegularVertexPatch : public VertexPatchBase<dim>
{
  /**
   * Type used to represent the orientation of a cell within a regular patch.
   */
  using CellOrientation = unsigned int;

  const static constexpr int dimension        = dim;
  const static bool          is_constant_size = true;
  const static unsigned int  n_cells          = 1 << dim;


  /**
   *  Constructor for a RegularVertexPatch.
   *
   * Orders the cells within the patch based on the central vertex index and
   * potentially generates orientation information. Asserts that the input
   * `patch` set contains exactly `n_cells`.
   *
   * @param patch A set of `CellIndex` objects representing the cells in the
   * patch.
   * @param vertex_index The global index of the central vertex.
   * @param index2cell A function object that converts a `CellIndex` to a
   * `CellIterator`.
   */
  RegularVertexPatch(
    const std::set<typename VertexPatchBase<dim>::CellIndex>                            &patch,
    const types::global_vertex_index                     &vertex_index,
    const std::function<typename VertexPatchBase<dim>::CellIterator(const typename VertexPatchBase<dim>::CellIndex &)> &index2cell);



  /**
   *  Returns the number of cells in the patch (always `n_cells`).
   */
  constexpr unsigned int
  size() const
  {
    return n_cells;
  }
  /**
   *  Checks if this patch conflicts with another patch for parallel
   * processing (e.g., based on shared DoFs).
   * @param other The other RegularVertexPatch to check against.
   * @return `true` if there is no conflict, `false` otherwise.
   */
  bool
  has_conflict_with(const RegularVertexPatch &other) const;


  /**
   *  Provides read-only access to the array of cell indices forming
   * the patch.
   * @return A constant reference to the array of `CellIndex`.
   */
  const auto &
  get_cells() const
  {
    return cells;
  }

  /**
   * Returns the cells in this patch as a vector.
   * Implements the pure virtual function from VertexPatchBase.
   *
   * @return A vector of CellIndex values representing the cells in the patch.
   */
  virtual std::vector<typename VertexPatchBase<dim>::CellIndex>
  get_cells_vector() const override
  {
    return std::vector<typename VertexPatchBase<dim>::CellIndex>(cells.begin(), cells.end());
  }


  /**
   *  Gets the orientation information for a specific cell within the
   * patch.
   * @param cell_index The local index of the cell within the patch (0 to
   * n_cells-1).
   * @return A constant reference to the `CellOrientation`.
   */
  const auto &
  get_orientation(const unsigned int &cell_index) const
  {
    AssertIndexRange(cell_index, size());
    return orientations[cell_index];
  }


  /**
   *  Static method to check if a given set of cells can form a
   * regular patch.
   *
   * Currently, this simply checks if the number of cells in the input `patch`
   * is equal to `n_cells`.
   *
   * @param patch A set of `CellIndex` objects.
   * @param vertex_index The global index of the potential central vertex
   * (unused).
   * @param index2cell A function object to convert `CellIndex` to
   * `CellIterator` (unused).
   * @return `true` if the patch can be constructed as a RegularVertexPatch,
   * `false` otherwise.
   */
  static bool
  is_constructible(
    const std::set<typename VertexPatchBase<dim>::CellIndex>                            &patch,
    const types::global_vertex_index                     &vertex_index,
    const std::function<typename VertexPatchBase<dim>::CellIterator(const typename VertexPatchBase<dim>::CellIndex &)> &index2cell)
  {
    (void)vertex_index;
    (void)index2cell;
    if (patch.size() == n_cells)
      return true;
    return false;
  }

private:
  std::array<typename VertexPatchBase<dim>::CellIndex, n_cells>       cells;
  std::array<CellOrientation, n_cells> orientations;
};

/**
 *  Represents a general patch, i.e., a patch that is not regular
 * (typically near boundaries or excluded vertices).
 *
 * Stores a variable number of cell indices.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
struct GeneralVertexPatch : public VertexPatchBase<dim>
{
  const static constexpr int dimension        = dim;
  const static bool          is_constant_size = false;

  /**
   *  Constructor for a GeneralVertexPatch.
   *
   * @param patch A set of `CellIndex` objects representing the cells in the
   * patch.
   * @param vertex_index The global index of the central vertex (unused in
   * current implementation).
   * @param index2cell A function object that converts a `CellIndex` to a
   * `CellIterator` (unused in current implementation).
   */
  GeneralVertexPatch(
    const std::set<typename VertexPatchBase<dim>::CellIndex>                            &patch,
    const types::global_vertex_index                         &vertex_index,
    const std::function<typename VertexPatchBase<dim>::CellIterator(const typename VertexPatchBase<dim>::CellIndex &)> &index2cell);

  /**
   *  Returns the number of cells in the patch.
   */
  unsigned int
  size() const
  {
    return cells.size();
  }

  /**
   *  Provides read-only access to the vector of cell indices forming
   * the patch.
   * @return A constant reference to the vector of `CellIndex`.
   */
  const auto &
  get_cells() const
  {
    return cells;
  }

  /**
   * Returns the cells in this patch as a vector.
   * Implements the pure virtual function from VertexPatchBase.
   *
   * @return A vector of CellIndex values representing the cells in the patch.
   */
  virtual std::vector<typename VertexPatchBase<dim>::CellIndex>
  get_cells_vector() const override
  {
    return cells;
  }

private:
  std::vector<typename VertexPatchBase<dim>::CellIndex> cells;
};

/**
 * Manages the storage and categorization of cell patches centered around
 * vertices.
 *
 * This class identifies patches of cells surrounding each vertex in a mesh
 * level managed by a MatrixFree object. It distinguishes between "regular"
 * patches (those with the expected number of cells, 2^dim) and "general"
 * patches (e.g., near boundaries). Patches can be categorized for parallel
 * processing, typically for patch-based smoothers like Gauss-Seidel.
 *
 * The class provides mechanisms to iterate over patches, access patch data,
 * and handle communication for parallel operations.
 *
 * @tparam MFType The type of the MatrixFree object used to access cell and DoF
 * information. Expected to provide types like `dimension`, `value_type`,
 * `vectorized_value_type`, and methods to access DoF handlers and cell
 * iterators.
 */
template <class MFType>
class PatchStorage : EnableObserverPointer
{
public:
  /**
   * Alias for the template parameter `MFType`.
   */
  using MatrixFreeType = MFType;


  /**
   * The spatial dimension.
   */
  const static constexpr unsigned int dim = MatrixFreeType::dimension;

  /**
   * The scalar value type used by the MatrixFree object (e.g., float,
   * double).
   */
  using value_type = typename MatrixFreeType::value_type;


  /**
   * The vectorized value type used by the MatrixFree object.
   */
  using vectorized_value_type = typename MatrixFreeType::vectorized_value_type;


  /**
   * The number of lanes in the vectorized type.
   */
  const static constexpr unsigned int n_lanes = vectorized_value_type::size();


  /**
   * The expected number of cells in a regular patch (2^dim).
   */
  const static constexpr unsigned int n_patch_cells = 1 << dim;


  /**
   * Type used to represent a unique index for each cell within the
   * context of the associated MatrixFree object. This is typically calculated
   * as `batch_index * n_lanes + lane_index`.
   */
  using CellIndex = unsigned int;


  /**
   * Alias for the cell iterator type of the underlying Triangulation.
   */
  using CellIterator = typename Triangulation<dim>::cell_iterator;

  /**
   * Type used to represent the orientation of a cell within a regular
   * patch (currently unused).
   */
  using CellOrientation = unsigned int;

  /**
   * Structure to hold additional data for patch generation, such as
   * vertices to exclude.
   */
  struct AdditionalData
  {
    /**
     * Enum for parallel task scheduling schemes.
     */
    enum TasksParallelScheme
    {
      none,
      by_color
    };

    /**
     * Default constructor. Initializes the excluded vertices set with
     * an invalid index and sets the default parallel scheme to 'none'.
     */
    AdditionalData()
      : excluded_vertices{numbers::invalid_unsigned_int}
      , tasks_parallel_scheme(none)
    {}

    /**
     * A set of global vertex indices that should be excluded during
     * patch generation.
     */
    std::set<types::global_vertex_index> excluded_vertices;

    /**
     * The parallel task scheduling scheme to use.
     * Default is 'none' (no parallelization).
     */
    TasksParallelScheme tasks_parallel_scheme;
  };


  /**
   * Alias for RegularVertexPatch with the current dimension.
   */
  using RegularPatch = RegularVertexPatch<dim>;

  /**
   * Alias for GeneralVertexPatch with the current dimension.
   */
  using GeneralPatch = GeneralVertexPatch<dim>;


  /**
   *  Alias for the task information type used for parallel scheduling
   * (e.g., Gauss-Seidel coloring). At the moment MPI is disabled.
   */
  using TaskInfoType = internal::GaussSeidel::TaskInfoDummy;
  // LinearAlgebra::GaussSeidel::TaskInfo<value_type, vectorized_value_type>;

  /**
   *  Type representing a range of patch indices [begin, end).
   */
  using PatchRange = const std::pair<std::size_t, std::size_t>;
  /**
   *  Type representing a category assigned to a patch, often used for
   * grouping or identification.
   */
  using PatchCategory = unsigned int;



  /**
   *  Constructor.
   * @param mf A shared pointer to the constant MatrixFree object.
   */
  PatchStorage(const std::shared_ptr<const MatrixFreeType> &mf);


  /**
   *  Initializes the PatchStorage by generating and storing patches.
   *
   * Identifies all vertex-centered patches on the specified level of the
   * MatrixFree object. Filters patches based on ownership and MPI rank.
   * Initializes internal data structures like partitioners and task info.
   *
   * @param data Additional data, e.g., vertices to exclude.
   */
  void
  initialize(const AdditionalData &data = AdditionalData());


  /**
   * Assigns a category to each regular patch based on a user-provided
   * function. @warning This function may change indices of patches.
   *
   * @param category_function A function that takes a `const RegularPatch &`
   * and returns a `PatchCategory`.
   */
  void
  categorize_patches(
    std::function<PatchCategory(const RegularPatch &)> category_function);


  /**
   *  Clears all stored patch data and resets the state to
   * uninitialized.
   */
  void
  clear();


  /**
   *  Executes a given function (`patch_worker`) over ranges of patches,
   * handling parallel communication.
   *
   * This function iterates through the parallel categories determined by the
   * `TaskInfoType`. For each category, it performs necessary communication
   * (e.g., updating ghost values) and then calls the `patch_worker` with the
   * range of regular patches belonging to that category.
   *
   * @tparam OutVector The type of the output/solution vector.
   * @tparam InVector The type of the input/rhs vector.
   * @param patch_worker The function to execute for each patch range. It
   * takes the `PatchStorage` instance, output vector, input vector, and the
   * `PatchRange` as arguments.
   * @param solution The output/solution vector. Its ghost values will be
   * updated during the loop.
   * @param rhs The input/rhs vector. Its ghost values must be up-to-date
   * before calling this function.
   * @param do_forward Flag indicating the direction of sweep (currently only
   * `true` is supported).
   */
  template <class OutVector, class InVector>
  void
  patch_loop(const std::function<void(const PatchStorage<MFType> &,
                                      OutVector &,
                                      const InVector &,
                                      const PatchRange &)> &patch_worker,
             OutVector                                     &solution,
             const InVector                                &rhs,
             const bool &do_forward = true) const;


  /**
   *  Helper function to adjusts the partitioner of a
   * distributed vector if it doesn't match the internal partitioner for the
   * given component.
   *
   * This ensures that ghost value communication works correctly within the
   * `patch_loop`. If the vector's partitioner is different, the vector is
   * reinitialized with the correct partitioner, preserving its locally owned
   * data.
   *
   * @tparam number The value type of the vector.
   * @param component The component index for which to check the partitioner.
   * @param vec The distributed vector to potentially adjust.
   */
  template <typename number>
  void
  adjust_ghost_range_if_necessary(
    const unsigned int                                component,
    const LinearAlgebra::distributed::Vector<number> &vec) const;


  /**
   * Gets a constant reference to the regular patch at the given global
   * index `i`. Used by FEPatchEvaluation
   *
   * The index `i` ranges from 0 to `n_patches() - 1`.
   * @param i The global index of the regular patch.
   * @return A constant reference to the `RegularPatch`.
   */
  const RegularPatch &
  get_regular_patch(const std::size_t &i) const;


  /**
   *  Gets the category assigned to the regular patch at the given global
   * index `i`.
   *
   * Requires `categorize_patches()` to have been called first.
   * @param i The global index of the regular patch.
   * @return A constant reference to the `PatchCategory`.
   */
  const PatchCategory &
  get_regular_patch_category(const std::size_t &i) const;


  /**
   *  Returns the total number of regular patches stored on the current
   * MPI process.
   */
  std::size_t
  n_patches() const;


  /**
   *  Gets a constant reference to the shared pointer holding the
   * MatrixFree object.
   */
  const std::shared_ptr<const MatrixFreeType> &
  get_matrix_free() const;


  /**
   *  Converts a `CellIndex` back to a `CellIterator`.
   * @param index The `CellIndex` to convert.
   * @return The corresponding `CellIterator`.
   */
  inline CellIterator
  index2cell(const CellIndex &index) const;


  /**
   *  Converts a MatrixFree batch index and lane index into a unique
   * `CellIndex`.
   * @param cell_batch_index The index of the cell batch in the MatrixFree
   * object.
   * @param lane_index The index of the lane within the batch.
   * @return The calculated `CellIndex`.
   */
  inline CellIndex
  batch2index(const unsigned int &cell_batch_index,
              const unsigned int &lane_index) const;


  /**
   *  Converts a `CellIndex` back into its corresponding MatrixFree batch
   * index and lane index.
   * @param cell_index The `CellIndex` to convert.
   * @return A pair containing the batch index (first) and lane index (second).
   */
  inline std::pair<unsigned int, unsigned int>
  index2batch(const CellIndex &cell_index) const;


  /**
   *  Outputs the geometry of all stored regular patches to VTU/PVTU
   * files.
   *
   * Each cell within each patch is written as a separate element in the VTU
   * file. Data associated with each cell includes patch index, parallel
   * category, local cell index within the patch, global cell index, MPI rank,
   * and user-defined category (if available).
   *
   * @param filename_without_extension The base name for the output files (e.g.,
   * "patches"). `.procXXXX.vtu` and `.pvtu` will be appended.
   */
  void
  output_patches(const std::string &filename_without_extension) const;


  /**
   *  Outputs the center points (approximated by one vertex) of all
   * stored regular patches to VTU/PVTU files.
   *
   * Each patch is represented by a single point (vertex) in the output file.
   * Data associated with each point includes patch index, parallel category,
   * MPI rank, and user-defined category (if available).
   *
   * @param filename_without_extension The base name for the output files (e.g.,
   * "patch_centers"). `.procXXXX.vtu` and `.pvtu` will be appended.
   */
  void
  output_centerpoints(const std::string &filename_without_extension) const;

private:
  std::shared_ptr<const MatrixFreeType> matrix_free;
  const Triangulation<dim>             &triangulation;

  const MPI_Comm     mpi_communicator;
  const unsigned int this_mpi_process;
  const unsigned int n_mpi_process;

  const unsigned int level;
  const unsigned int n_components;



  std::map<types::global_vertex_index, std::set<CellIndex>>
  generate_patches();


  void
  push_back_patch(const std::set<CellIndex>        &patch,
                  const types::global_vertex_index &vertex_index);


  inline std::set<types::global_dof_index>
  collect_patch_dof_indices(const std::set<CellIndex> &patch_cells,
                            const unsigned int        &component) const;


  /**
   * Colorize patches using graph coloring to enable parallel processing.
   *
   * This method builds a graph where each patch is a node, and edges connect
   * patches that have overlapping cells. Graph coloring is then performed
   * using deal.II's GraphColoring::make_graph_coloring function to assign
   * colors such that no two patches with the same color share cells.
   *
   * @param parallel_cat The parallel category for which to colorize patches.
   *                     Currently unused but kept for future extension.
   * @return A vector of colors, one for each patch in the given category.
   */
  std::vector<unsigned int>
  colorize_patches(unsigned int parallel_cat);


  // patches[parallel cat][patch_index]
  std::array<std::vector<RegularPatch>, TaskInfoType::n_categories>

    regular_patches;
  std::array<std::vector<PatchCategory>, TaskInfoType::n_categories>
    regular_patch_categories;
  std::array<std::vector<GeneralPatch>, TaskInfoType::n_categories>
    other_patches;



  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  std::vector<unsigned int>         process_colors;
  mutable std::vector<TaskInfoType> task_infos;


  AdditionalData additional_data;

  bool is_initialized;
  bool are_patches_categorized;
};


#ifndef DOXYGEN
//---------------------------------------------------------------------------

// ============================================================================
// Inline member function definitions
// ============================================================================

template <class MFType>
inline typename PatchStorage<MFType>::CellIterator
PatchStorage<MFType>::index2cell(const CellIndex &index) const
{
  const unsigned int cell_batch_index = index / n_lanes;
  const unsigned int lane_index       = index % n_lanes;
  return matrix_free->get_cell_iterator(cell_batch_index, lane_index);
}


template <class MFType>
inline typename PatchStorage<MFType>::CellIndex
PatchStorage<MFType>::batch2index(const unsigned int &cell_batch_index,
                                  const unsigned int &lane_index) const
{
  return cell_batch_index * n_lanes + lane_index;
}


template <class MFType>
inline std::pair<unsigned int, unsigned int>
PatchStorage<MFType>::index2batch(const CellIndex &cell_index) const
{
  const unsigned int cell_batch_index = cell_index / n_lanes;
  const unsigned int lane_index       = cell_index % n_lanes;
  return std::make_pair(cell_batch_index, lane_index);
}


template <class MFType>
template <typename number>
void
PatchStorage<MFType>::adjust_ghost_range_if_necessary(
  const unsigned int                                component,
  const LinearAlgebra::distributed::Vector<number> &vec) const
{
  if (vec.get_partitioner().get() == partitioners[component].get())
    return;
  LinearAlgebra::distributed::Vector<number> copy_vec(vec);
  const_cast<LinearAlgebra::distributed::Vector<number> &>(vec).reinit(
    partitioners[component]);
  const_cast<LinearAlgebra::distributed::Vector<number> &>(vec)
    .copy_locally_owned_data_from(copy_vec);
}


template <class MFType>
template <class OutVector, class InVector>
inline void
PatchStorage<MFType>::patch_loop(
  const std::function<void(const PatchStorage<MFType> &,
                           OutVector &,
                           const InVector &,
                           const PatchRange &)> &patch_worker,
  OutVector                                     &solution,
  const InVector                                &rhs,
  const bool                                    &do_forward) const
{
  Assert(is_initialized, ExcNotInitialized());
  Assert(do_forward == true, ExcNotImplemented());
  (void)do_forward;

  adjust_ghost_range_if_necessary(0, rhs);
  adjust_ghost_range_if_necessary(0, solution);

  rhs.update_ghost_values();
  solution.set_ghost_state(true);

  solution.update_ghost_values();

  std::size_t current_begin = 0;


  for (unsigned int cat = 0; cat < TaskInfoType::n_categories; ++cat)
    {
      for (unsigned int component = 0; component < n_components; ++component)
        {
          const int communication_channel = component;
          task_infos[component].communicate(cat,
                                            communication_channel,
                                            solution);
        }
      if (regular_patches[cat].size() != 0)
        {
          PatchRange patch_range(current_begin,
                                 current_begin + regular_patches[cat].size());
          patch_worker(*this, solution, rhs, patch_range);

          current_begin += regular_patches[cat].size();
        }
    }

  for (unsigned int component = 0; component < n_components; ++component)
    {
      const int communication_channel = component;
      task_infos[component].communicate(TaskInfoType::n_categories,
                                        communication_channel,
                                        solution);
    }


  solution.zero_out_ghost_values();
}

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE



#endif // dealii__patch_storage_h
