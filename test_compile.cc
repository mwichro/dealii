#include <deal.II/matrix_free/patch_storage.h>
#include <deal.II/matrix_free/matrix_free.h>

// Verify we can instantiate the new types
template struct RegularVertexPatch<dealii::MatrixFree<2, double>>;
template struct GeneralVertexPatch<dealii::MatrixFree<2, double>>;

int main() {
    return 0;
}
