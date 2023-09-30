#include <iostream>
#include "tensor_logic_shape.h"
namespace wjy {

class Tensor {
   private:
    void* const data_ = nullptr;
    int64_t element_numbers = 0;
    TensorLogicShape shape;

   protected:
    int element_size_ = -1;

   public:
    Tensor(/* args */);
    ~Tensor();
    int64_t size() { return element_numbers; }
    int64_t element_size() { return element_size_; }
    void* const data() { return data_; }
    int64_t stride(int64_t dim) { return 0; }
    int64_t dim_size(int64_t dim) { return shape.dims()[dim]; }
    std::vector<int64_t>& dims() { return shape.dims(); }
    Strids strides() { return shape.strides(); }
    // 有待商榷
    template <class T>
    T* values() {
        return reinterpret_cast<T*>(data_);
    }
};

}  // namespace wjy