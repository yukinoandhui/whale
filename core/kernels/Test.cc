#include "tensor/tensor.h"
namespace kernels {

template <class T>
void tensor_equal_vector(Tensor output, std::vector<T>) {}

#define TENSOR_EQUAL_VECTOR()

int main() {
    return 0;
}
};  // namespace kernels