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



#include <deal.II/matrix_free/patch_storage.h>
#include <fstream>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  /*
   * apply permutation to the given vector.
   */
  template <class T>
  void
  reorder(std::vector<T> &vA, std::vector<size_t> vOrder)
  {
    AssertDimension(vA.size(), vOrder.size());

    // for all elements to put in place
    for (size_t i = 0; i < vA.size(); ++i)
      {
        // while vOrder[i] is not yet in place
        // every swap places at least one element in its proper place
        while (vOrder[i] != vOrder[vOrder[i]])
          {
            std::swap(vA[vOrder[i]], vA[vOrder[vOrder[i]]]);
            std::swap(vOrder[i], vOrder[vOrder[i]]);
          }
      }
  }



  // Computes index of given vertex within a cell
  template <int dim>
  unsigned int
  compute_vertex_index(const typename Triangulation<dim>::cell_iterator &cell,
                       const types::global_vertex_index                 &vertex)
  {
    for (const auto &i : GeometryInfo<dim>::vertex_indices())
      if (cell->vertex_index(i) == vertex)
        return i;
    Assert(false, ExcMessage("Vertex not found on the given cell!"));
    return numbers::invalid_unsigned_int;
  }

  // helper functions for RegularVertexPatch constructor
  template <int dim>
  std::vector<typename Triangulation<dim>::cell_iterator>
  order_patch(
    const std::set<typename Triangulation<dim>::cell_iterator> &patch_cells,
    const types::global_vertex_index                           &vertex_index);

  // rotate patch to minimize cell rotations
  void
  orient_patch2D(
    std::vector<typename Triangulation<2>::cell_iterator> &patch_cells,
    const types::global_vertex_index                      &vertex);

  template <int dim>
  void
  rotate_patch(
    std::vector<typename Triangulation<dim>::cell_iterator> &patch_cells);


  // Explicit specialization declaration BEFORE any use
  template <>
  void
  rotate_patch<2>(
    std::vector<typename Triangulation<2>::cell_iterator> &patch_cells);


  void
  orient_patch2D(
    std::vector<typename Triangulation<2>::cell_iterator> &patch_cells,
    const types::global_vertex_index                      &vertex)
  {
    const static unsigned int lookup_rotations[4] = {0, 3, 1, 2};
    const unsigned int        n_rotations =
      lookup_rotations[internal::compute_vertex_index<2>(patch_cells[0],
                                                         vertex)];
    for (unsigned int i = 0; i < n_rotations; ++i)
      rotate_patch<2>(patch_cells);
  }

  template <>
  std::vector<typename Triangulation<2>::cell_iterator>
  order_patch<2>(
    const std::set<typename Triangulation<2>::cell_iterator> &patch_cells,
    const types::global_vertex_index                         &vertex_index)
  {
    const constexpr int                 dim = 2;
    const static constexpr unsigned int n_vertices =
      GeometryInfo<dim>::vertices_per_cell;


    // in case of non-standard patch just copy cell into vector
    if (patch_cells.size() != n_vertices)
      return std::vector<typename Triangulation<2>::cell_iterator>(
        patch_cells.begin(), patch_cells.end());


    std::set<typename Triangulation<dim>::cell_iterator> patch_copy =
      patch_cells;
    // we have a regular patch, thus  we need to order the cells:
    std::vector<typename Triangulation<dim>::cell_iterator> ordered_cells;
    ordered_cells.reserve(n_vertices);
    ordered_cells.push_back(*patch_copy.begin());
    patch_copy.erase(patch_copy.begin());


    // Look-up table. If face i is on the left of the cell
    // right2upper[i] tells which face is on the upper side of cell
    const constexpr std::array<int, n_vertices> right2upper = {{3, 2, 0, 1}};

    for (unsigned int i = 0; i < n_vertices; ++i)
      {
        if (patch_copy.find(ordered_cells[0]->neighbor(i)) !=
              patch_copy.end() &&
            patch_copy.find(ordered_cells[0]->neighbor(right2upper[i])) !=
              patch_copy.end())
          {
            ordered_cells.push_back(ordered_cells[0]->neighbor(right2upper[i]));
            patch_copy.erase(ordered_cells[0]->neighbor(right2upper[i]));

            ordered_cells.push_back(ordered_cells[0]->neighbor(i));
            patch_copy.erase(ordered_cells[0]->neighbor(i));
          }
      }

    AssertDimension(patch_copy.size(), 1);

    ordered_cells.push_back(*patch_copy.begin());

    orient_patch2D(ordered_cells, vertex_index);
    return ordered_cells;
  }


  // counter-clockwise rotation of the given patch
  template <>
  void
  rotate_patch<2>(
    std::vector<typename Triangulation<2>::cell_iterator> &patch_cells)
  {
    static const std::vector<size_t> patch_rotation = {2, 0, 3, 1};

    internal::reorder(patch_cells, patch_rotation);
  }


  namespace GaussSeidel
  {
    unsigned int
    min_category(const unsigned int a, const unsigned int b)
    {
      if ((a == 1 && b == 2) && (b == 1 && a == 2))
        return 0;

      return std::min(a, b);
    }
  } // namespace GaussSeidel

} // namespace internal


// ============================================================================
// RegularVertexPatch constructor implementations
// ============================================================================

template <int dim>
RegularVertexPatch<dim>::RegularVertexPatch(
  const std::set<CellIndex>                            &patch,
  const types::global_vertex_index                     &vertex_index,
  const std::function<CellIterator(const CellIndex &)> &index2cell)
{
  if constexpr (dim == 2)
    {
      std::map<CellIterator, CellIndex> iterator2index;
      std::set<CellIterator>            cells_iterators;

      for (auto &cell_index : patch)
        {
          iterator2index[index2cell(cell_index)] = cell_index;
          cells_iterators.insert(index2cell(cell_index));
        }

      std::vector<CellIterator> ordered_patch_cells =
        internal::order_patch<dim>(cells_iterators, vertex_index);

      internal::reorder(ordered_patch_cells, {3, 2, 1, 0});

      std::vector<CellIndex> ordered_patch;
      for (auto &cell : ordered_patch_cells)
        ordered_patch.push_back(iterator2index.at(cell));


      AssertDimension(ordered_patch.size(), ordered_patch_cells.size());

      for (unsigned int i = 0; i < n_cells; ++i)
        cells[i] = ordered_patch[i];


      this->partially_ghosted = false;
      for (auto &cell : ordered_patch_cells)
        if (!cell->is_locally_owned_on_level())
          this->partially_ghosted = true;
    }

  if constexpr (dim == 3)
    {
      const static std::array<std::size_t, n_cells> vindex2position = {
        {7, 6, 5, 4, 3, 2, 1, 0}};

      std::vector<bool> used_cell(n_cells, false);

      for (const auto cell_index : patch)
        {
          const auto cell = index2cell(cell_index);
          const auto v_index =
            internal::compute_vertex_index<dim>(cell, vertex_index);

          const auto &cell__inpatch_index = vindex2position[v_index];
          Assert(
            used_cell[cell__inpatch_index] == false,
            ExcMessage(
              "You have tried to construct patch from rotated cells, that is currently not implemented"));
          used_cell[cell__inpatch_index] = true;
          cells[cell__inpatch_index]     = cell_index;
        }

      this->partially_ghosted = false;
      for (auto &cell_index : cells)
        if (!index2cell(cell_index)->is_locally_owned_on_level())
          this->partially_ghosted = true;
    }
}


template <int dim>
bool
RegularVertexPatch<dim>::has_conflict_with(
  const RegularVertexPatch &other) const
{
  (void)other;
  return false;
}


// ============================================================================
// GeneralVertexPatch constructor implementation
// ============================================================================

template <int dim>
GeneralVertexPatch<dim>::GeneralVertexPatch(
  const std::set<CellIndex> &patch,
  const types::global_vertex_index & /*vertex_index*/,
  const std::function<CellIterator(const CellIndex &)> &index2cell)
{
  cells.assign(patch.begin(), patch.end());

  // Check if any cells are ghosted
  this->partially_ghosted = false;
  for (const auto &cell_index : cells)
    if (!index2cell(cell_index)->is_locally_owned_on_level())
      this->partially_ghosted = true;
}


// ============================================================================
// PatchStorage member function implementations
// ============================================================================

template <class MFType>
PatchStorage<MFType>::PatchStorage(
  const std::shared_ptr<const MatrixFreeType> &mf)
  : EnableObserverPointer()
  , matrix_free(mf)
  , triangulation(matrix_free->get_dof_handler().get_triangulation())
  , mpi_communicator(
      matrix_free->get_vector_partitioner()->get_mpi_communicator())
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , n_mpi_process(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , level(matrix_free->get_mg_level())
  , n_components(matrix_free->n_components())
  , task_infos(n_components)
  , is_initialized(false)
  , are_patches_categorized(false)
{
  Assert(matrix_free->get_dof_handler().get_triangulation().n_global_levels() >
           0,
         ExcInternalError());
  Assert(level != numbers::invalid_unsigned_int, ExcInternalError());
}


template <class MFType>
void
PatchStorage<MFType>::initialize(const AdditionalData &data)
{
  this->additional_data = data;

  Assert(static_cast<unsigned int>(level) < triangulation.n_global_levels(),
         ExcInternalError());

  std::map<types::global_vertex_index, std::set<CellIndex>> vertex_to_cell_map =
    generate_patches();

  {
    unsigned int n_patches = 0;
    for (auto iterator = vertex_to_cell_map.begin();
         iterator != vertex_to_cell_map.end();
         ++iterator)
      {
        std::set<CellIndex> &patch = iterator->second;

        std::set<CellIterator> patch_cells;
        for (const CellIndex &cells_indices : patch)
          patch_cells.insert(index2cell(cells_indices));


        bool         has_owned = false;
        unsigned int owned_by  = numbers::invalid_unsigned_int;
        for (const CellIterator &cell : patch_cells)
          {
            if (cell->is_locally_owned_on_level())
              {
                has_owned = true;
              }
            if (cell->level_subdomain_id() < owned_by &&
                cell->level_subdomain_id() <= n_mpi_process)
              {
                owned_by = cell->level_subdomain_id();
              }
          }
        if (false == has_owned ||
            Utilities::MPI::this_mpi_process(mpi_communicator) != owned_by)
          patch.clear();
        else
          ++n_patches;
      }
    Assert(n_patches <= triangulation.n_vertices(), ExcInternalError());
  }

  for (unsigned int component = 0; component < n_components; ++component)
    {
      const auto &dof_handler = matrix_free->get_dof_handler(component);

      IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);

      std::set<types::global_dof_index> locally_relevant_dofs_set;
      for (auto [vertex_index, patch] : vertex_to_cell_map)
        {
          const auto local_dofs = collect_patch_dof_indices(patch, component);
          locally_relevant_dofs_set.insert(local_dofs.begin(),
                                           local_dofs.end());
        }

      IndexSet dependency_region(dof_handler.n_dofs());
      dependency_region.add_indices(locally_relevant_dofs_set.begin(),
                                    locally_relevant_dofs_set.end());

      IndexSet influence_region = dependency_region;

      partitioners.push_back(std::make_shared<Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_mg_dofs(level),
        locally_relevant_dofs,
        mpi_communicator));

      task_infos[component].initialize(*matrix_free,
                                       dependency_region,
                                       influence_region,
                                       process_colors,
                                       true);
    }

  for (auto iterator = vertex_to_cell_map.begin();
       iterator != vertex_to_cell_map.end();
       ++iterator)
    {
      std::set<CellIndex>              &patch        = iterator->second;
      const types::global_vertex_index &vertex_index = iterator->first;
      const unsigned int                n_cells      = patch.size();

      if (n_cells == 0)
        continue;

      push_back_patch(patch, vertex_index);
    }

  for (unsigned int i = 0; i < TaskInfoType::n_categories; ++i)
    std::sort(regular_patches[i].begin(),
              regular_patches[i].end(),
              [&](const auto a, const auto b) {
                return a.get_cells() < b.get_cells();
              });

  is_initialized = true;
}


template <class MFType>
std::map<types::global_vertex_index,
         std::set<typename PatchStorage<MFType>::CellIndex>>
PatchStorage<MFType>::generate_patches()
{
  std::map<types::global_vertex_index, std::set<CellIndex>> vertex_to_cell_map;

  for (unsigned int i = 0;
       i < matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
       ++i)
    for (unsigned int j = 0;
         j < matrix_free->n_active_entries_per_cell_batch(i);
         ++j)
      {
        const auto &cell = matrix_free->get_cell_iterator(i, j);

        for (const auto v : cell->vertex_indices())
          vertex_to_cell_map[cell->vertex_index(v)].insert(batch2index(i, j));
      }

  return vertex_to_cell_map;
}


template <class MFType>
void
PatchStorage<MFType>::push_back_patch(
  const std::set<typename PatchStorage<MFType>::CellIndex> &patch,
  const types::global_vertex_index                         &vertex_index)
{
  const std::function<CellIterator(const CellIndex &)> index2cell_local =
    [&](const CellIndex &index) -> CellIterator {
    return this->index2cell(index);
  };

  unsigned int category = TaskInfoType::invalid_category;
  for (unsigned int component = 0; component < n_components; ++component)
    {
      const auto local_dofs = collect_patch_dof_indices(patch, component);

      category = internal::GaussSeidel::min_category(
        category,
        task_infos[component].determine_minimum_category_of_patch(
          local_dofs, process_colors));
    }
  const unsigned int partition =
    TaskInfoType::determine_cell_partition(category);

  if (RegularPatch::is_constructible(patch, vertex_index, index2cell_local))
    {
      RegularPatch ordered_patch(patch, vertex_index, index2cell_local);
      regular_patches[partition].push_back(ordered_patch);
    }
  else
    {
      GeneralPatch general_patch(patch, vertex_index, index2cell_local);
      other_patches[partition].push_back(general_patch);
    }
}


template <class MFType>
std::set<types::global_dof_index>
PatchStorage<MFType>::collect_patch_dof_indices(
  const std::set<typename PatchStorage<MFType>::CellIndex> &patch_cells,
  const unsigned int                                       &component) const
{
  const auto &dof_handler = matrix_free->get_dof_handler(component);
  std::set<types::global_dof_index> local_dofs;

  for (auto cell_index : patch_cells)
    {
      const auto &cell = index2cell(cell_index);
      typename DoFHandler<dim>::level_cell_iterator dof_cell(&triangulation,
                                                             cell->level(),
                                                             cell->index(),
                                                             &dof_handler);

      std::vector<types::global_dof_index> local_dof_indices(
        dof_cell->get_fe().n_dofs_per_cell());
      dof_cell->get_mg_dof_indices(local_dof_indices);
      local_dofs.insert(local_dof_indices.begin(), local_dof_indices.end());
    }
  return local_dofs;
}


template <class MFType>
void
PatchStorage<MFType>::categorize_patches(
  std::function<PatchCategory(const RegularPatch &)> category_function)
{
  Assert(is_initialized, ExcNotInitialized());

  for (unsigned int cat = 0; cat < TaskInfoType::n_categories; ++cat)
    {
      regular_patch_categories[cat].reserve(regular_patches[cat].size());
      for (const auto &patch : regular_patches[cat])
        {
          regular_patch_categories[cat].push_back(category_function(patch));
        }
    }
  are_patches_categorized = true;
}


template <class MFType>
void
PatchStorage<MFType>::clear()
{
  for (unsigned int i = 0; i < TaskInfoType::n_categories; ++i)
    {
      regular_patches[i].clear();
      regular_patch_categories[i].clear();
      other_patches[i].clear();
    }
  partitioners.clear();
  process_colors.clear();
  task_infos.clear();
  task_infos.resize(n_components);
  is_initialized          = false;
  are_patches_categorized = false;
}


template <class MFType>
const typename PatchStorage<MFType>::RegularPatch &
PatchStorage<MFType>::get_regular_patch(const std::size_t &i) const
{
  std::size_t ii = i;
  for (unsigned int cat_index = 0; cat_index < regular_patches.size();
       ++cat_index)
    {
      const auto &patch_cat = regular_patches[cat_index];
      if (ii < patch_cat.size())
        return patch_cat[ii];
      else
        ii -= patch_cat.size();
    }

  Assert(false, ExcInternalError());
  return regular_patches[0][0];
}


template <class MFType>
std::size_t
PatchStorage<MFType>::n_patches() const
{
  std::size_t n_patches = 0;
  for (const auto &patch_cat : regular_patches)
    n_patches += patch_cat.size();
  return n_patches;
}


template <class MFType>
const std::shared_ptr<const typename PatchStorage<MFType>::MatrixFreeType> &
PatchStorage<MFType>::get_matrix_free() const
{
  return matrix_free;
}


template <class MFType>
const typename PatchStorage<MFType>::PatchCategory &
PatchStorage<MFType>::get_regular_patch_category(const std::size_t &i) const
{
  Assert(are_patches_categorized, ExcNotInitialized());
  AssertDimension(regular_patches.size(), regular_patch_categories.size());

  std::size_t ii = i;
  for (unsigned int cat_index = 0; cat_index < regular_patches.size();
       ++cat_index)
    {
      AssertDimension(regular_patches[cat_index].size(),
                      regular_patch_categories[cat_index].size());
      const auto &patch_cat = regular_patch_categories[cat_index];
      if (ii < patch_cat.size())
        return patch_cat[ii];
      else
        ii -= patch_cat.size();
    }

  AssertThrow(false, ExcInternalError());
  return regular_patch_categories[0][0];
}


template <class MFType>
void
PatchStorage<MFType>::output_patches(
  const std::string &filename_without_extension) const
{
  using CellOutData = DataOutBase::Patch<dim, dim>;
  std::vector<CellOutData> patches_out;

  std::vector<std::string> data_names;
  data_names.emplace_back("patch_index");
  data_names.emplace_back("parallel category");
  data_names.emplace_back("local cell index");
  data_names.emplace_back("global cell index");
  data_names.emplace_back("MPIRank");
  data_names.emplace_back("Category");

  const unsigned n_datasets          = 6;
  unsigned int   patch_counter       = 0;
  unsigned int   patch_index_counter = 0;
  for (unsigned int cat_index = 0; cat_index < regular_patches.size();
       ++cat_index)
    {
      const auto &patch_cat = regular_patches[cat_index];
      for (unsigned int patch_index = 0; patch_index < patch_cat.size();
           ++patch_index)
        {
          const auto &patch = patch_cat[patch_index];
          for (unsigned int cell_index_within_patch = 0;
               cell_index_within_patch < patch.get_cells().size();
               ++cell_index_within_patch)
            {
              const auto &cell_index =
                patch.get_cells()[cell_index_within_patch];
              const auto &cell = index2cell(cell_index);
              CellOutData cell_out;
              for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
                   ++i)
                cell_out.vertices[i] = cell->vertex(i);
              cell_out.patch_index    = patch_counter;
              cell_out.reference_cell = cell->reference_cell();
              cell_out.data.reinit(n_datasets, 1 << dim);
              for (unsigned int i = 0; i < 1 << dim; ++i)
                {
                  cell_out.data(0, i) = patch_index_counter;
                  cell_out.data(1, i) = cat_index;
                  cell_out.data(2, i) = cell_index_within_patch;
                  cell_out.data(3, i) = cell_index;
                  cell_out.data(4, i) = this_mpi_process;
                  if (are_patches_categorized)
                    cell_out.data(5, i) =
                      get_regular_patch_category(patch_index_counter);
                  else
                    cell_out.data(5, i) =
                      std::numeric_limits<double>::quiet_NaN();
                }

              cell_out.n_subdivisions = 1;
              patches_out.push_back(cell_out);
              patch_counter++;
            }
          ++patch_index_counter;
        }
    }

  std::vector<std::string> piece_names(n_mpi_process);
  for (unsigned int i = 0; i < n_mpi_process; ++i)
    piece_names[i] = filename_without_extension + ".proc" +
                     Utilities::int_to_string(i, 4) + ".vtu";
  std::string new_file = piece_names[this_mpi_process];

  std::string out_pvtu = filename_without_extension + ".pvtu";

  std::ofstream out(new_file);
  std::vector<
    std::tuple<unsigned int,
               unsigned int,
               std::string,
               DataComponentInterpretation::DataComponentInterpretation>>
    vector_data_ranges;

  DataOutBase::VtkFlags vtu_flags;

  DataOutBase::write_vtu(
    patches_out, data_names, vector_data_ranges, vtu_flags, out);

  if (this_mpi_process == 0)
    {
      std::ofstream pvtu_output(out_pvtu);
      std::ostream &pvtu_out_steam = pvtu_output;
      DataOutBase::write_pvtu_record(
        pvtu_out_steam, piece_names, data_names, vector_data_ranges, vtu_flags);
    }
}


template <class MFType>
void
PatchStorage<MFType>::output_centerpoints(
  const std::string &filename_without_extension) const
{
  using PointOutData = DataOutBase::Patch<0, dim>;
  std::vector<PointOutData> centerpoints_out;

  std::vector<std::string> data_names;
  data_names.emplace_back("patch_index");
  data_names.emplace_back("parallel category");
  data_names.emplace_back("MPIRank");
  data_names.emplace_back("Category");

  const unsigned n_datasets    = 4;
  unsigned int   patch_counter = 0;

  for (unsigned int cat_index = 0; cat_index < regular_patches.size();
       ++cat_index)
    {
      const auto &patch_cat = regular_patches[cat_index];
      for (unsigned int patch_index = 0; patch_index < patch_cat.size();
           ++patch_index)
        {
          const auto        &patch                   = patch_cat[patch_index];
          const unsigned int cell_index_within_patch = 0;
          const auto  &cell_index = patch.get_cells()[cell_index_within_patch];
          const auto  &cell       = index2cell(cell_index);
          PointOutData vertex_out;

          unsigned int last_vertex_index =
            GeometryInfo<dim>::vertices_per_cell - 1;

          vertex_out.vertices[0] = cell->vertex(last_vertex_index);

          vertex_out.patch_index = patch_counter;

          vertex_out.data.reinit(n_datasets, 1);

          vertex_out.data(0, 0) = patch_counter;
          vertex_out.data(1, 0) = cat_index;
          vertex_out.data(2, 0) = this_mpi_process;
          if (are_patches_categorized)
            vertex_out.data(3, 0) = get_regular_patch_category(patch_counter);
          else
            vertex_out.data(3, 0) = std::numeric_limits<double>::quiet_NaN();

          centerpoints_out.push_back(vertex_out);
          ++patch_counter;
        }
    }

  std::vector<std::string> piece_names(n_mpi_process);
  for (unsigned int i = 0; i < n_mpi_process; ++i)
    piece_names[i] = filename_without_extension + ".proc" +
                     Utilities::int_to_string(i, 4) + ".vtu";
  std::string new_file = piece_names[this_mpi_process];

  std::string out_pvtu = filename_without_extension + ".pvtu";


  std::ofstream out(new_file);
  std::vector<
    std::tuple<unsigned int,
               unsigned int,
               std::string,
               DataComponentInterpretation::DataComponentInterpretation>>
    vector_data_ranges;

  DataOutBase::VtkFlags vtu_flags;

  DataOutBase::write_vtu(
    centerpoints_out, data_names, vector_data_ranges, vtu_flags, out);
  if (this_mpi_process == 0)
    {
      std::ofstream pvtu_output(out_pvtu);
      std::ostream &pvtu_out_steam = pvtu_output;
      DataOutBase::write_pvtu_record(
        pvtu_out_steam, piece_names, data_names, vector_data_ranges, vtu_flags);
    }
}


template <class MFType>
std::vector<unsigned int>
PatchStorage<MFType>::colorize_patches(unsigned int parallel_cat)
{
  Assert(is_initialized, ExcNotInitialized());

  // Work only on the requested parallel category
  AssertIndexRange(parallel_cat, regular_patches.size());

  const auto        &patch_cat       = regular_patches[parallel_cat];
  const unsigned int n_patches_total = patch_cat.size();

  if (n_patches_total == 0)
    return std::vector<unsigned int>();

  // Build a mapping from patch address to index for O(1) lookup later.
  std::map<const RegularPatch *, unsigned int> patch_to_index;
  for (unsigned int i = 0; i < n_patches_total; ++i)
    patch_to_index[&patch_cat[i]] = i;

  // Conflict function for graph coloring. Two patches conflict if they
  // share any cell index. The GraphColoring API expects a function that
  // maps an iterator to a vector of indices that indicate conflicts.
  auto get_conflict_indices =
    [&](const typename std::vector<RegularPatch>::const_iterator &patch_it)
    -> std::vector<types::global_dof_index> {
    const RegularPatch &patch        = *patch_it;
    const auto          cells_vector = patch.get_cells_vector();

    std::vector<types::global_dof_index> conflict_indices;
    conflict_indices.reserve(cells_vector.size());
    for (const auto &cell_idx : cells_vector)
      conflict_indices.push_back(
        static_cast<types::global_dof_index>(cell_idx));
    return conflict_indices;
  };

  // Perform graph coloring on the patches of the selected category.
  auto coloring = GraphColoring::make_graph_coloring(patch_cat.cbegin(),
                                                     patch_cat.cend(),
                                                     get_conflict_indices);

  // Convert coloring result to a vector of colors per patch in this category
  std::vector<unsigned int> patch_colors(n_patches_total);
  for (unsigned int color = 0; color < coloring.size(); ++color)
    for (const auto &patch_it : coloring[color])
      {
        const RegularPatch *patch_ptr   = &*patch_it;
        const unsigned int  patch_index = patch_to_index.at(patch_ptr);
        patch_colors[patch_index]       = color;
      }

  return patch_colors;
}


// Explicit template instantiations
template class VertexPatchBase<2>;
template class VertexPatchBase<3>;

template struct RegularVertexPatch<2>;
template struct RegularVertexPatch<3>;

template struct GeneralVertexPatch<2>;
template struct GeneralVertexPatch<3>;

template class PatchStorage<MatrixFree<2, double>>;
template class PatchStorage<MatrixFree<3, double>>;

template class PatchStorage<MatrixFree<2, float>>;
template class PatchStorage<MatrixFree<3, float>>;


DEAL_II_NAMESPACE_CLOSE
