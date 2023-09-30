#include <stdint.h>
#include <vector>
namespace wjy {

using Strids = std::vector<int64_t>;
using Dims = std::vector<int64_t>;

// 只有3d或者3d以上才有layout
class Layout {
   private:
    int64_t batch;
    int64_t channel;
    std::vector<int64_t> spital_dimensions;

   public:
    Layout(int64_t batch,
           int64_t channel,
           std::vector<int64_t> spital_dimensions)
        : batch(batch),
          channel(channel),
          spital_dimensions(spital_dimensions) {}
    // 默认是nchw
    Layout nchw_layout() { return Layout(batch, channel, spital_dimensions); }
    Layout nhwc_layout() {}
    ~Layout();
};

class TensorLogicShape {
   private:
    Strids strides_;
    Dims dims_;

   protected:
    int element_size = -1;

   public:
    TensorLogicShape(/* args */);
    ~TensorLogicShape();
    Strids& strides() { return strides_; }
    Dims& dims() { return dims_; }
};

}  // namespace wjy