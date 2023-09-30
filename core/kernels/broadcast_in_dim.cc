#include <cstring>
#include <iostream>
#include <vector>
#include "kernel_helper.h"
using namespace std;
namespace wjy {
namespace kernels {
template <class T>
void Broadcast(T* output, int cur_len, int target_len, int element_size) {
    int copy_len = target_len - cur_len > 0 ? cur_len : target_len - cur_len;
    while (copy_len > 0) {
        memcpy(output + cur_len, output, copy_len * element_size);
        cur_len += copy_len;
        copy_len += copy_len;
        copy_len =
            target_len - cur_len >= copy_len ? copy_len : target_len - cur_len;
    }
}

template <class T>
void BroadcastInDim(vector<T>& input,
                    vector<T>& output,
                    vector<int>& broadcast_dimensions) {
    auto element_size = sizeof(T);  // 后续可以用tensor替换
    if (input.size() == 1) {
        braodcast(input.data(), 1, output.size(), element_size);
    } else if (input.size() == output.size()) {
        memcpy(output.data(), input.data(), input.size() * element_size);
    } else {
    }
}

template <class T>
void broadcast_recursive(int cur_idx,
                         int last,
                         T* input,
                         T* output,
                         std::vector<int64_t>& cur2last_dim_mul,
                         int& input_offset,
                         int& output_offset,
                         std::vector<int64_t>& output_dims,
                         std::vector<int64_t>& input_dims,
                         int64_t element_size) {
    if (cur_idx == last) {
        // 假设最后一个维度是相等的,那么其实就是直接copy
        // 如果不相等，那么其实就是copy一个然后broadcast

        memcpy(output + output_offset, input + input_offset,
               input_dims[cur_idx] * element_size);
        input_offset += input_dims[cur_idx];
        output_offset += input_dims[cur_idx];
        if (output_dims[last] != input_dims[last]) {
            Broadcast(output, input_dims[cur_idx], output_dims[cur_idx],
                      element_size);
            output_offset += output_dims[cur_idx] - 1;
        }
        return;
    }
    if (input_dims[cur_idx] != output_dims[cur_idx]) {
        broadcast_recursive(cur_idx + 1, last, input, output, cur2last_dim_mul,
                            input_offset, output_offset, output_dims,
                            input_dims, element_size);
        Broadcast(output, cur2last_dim_mul[cur_idx + 1],
                  cur2last_dim_mul[cur_idx], element_size);
        output_offset +=
            cur2last_dim_mul[cur_idx] - cur2last_dim_mul[cur_idx + 1];
        return;
    }
    // 相等的情况，此时就比较复杂了，不管后面的维度是否经历了广播，我们都要for循环处理
    /*
        broadcast_recursive(cur_idx + 1, last, input, output, cur2last_dim_mul,
                        input_offset, output_offset, output_dims, input_dims,
                        element_size);
     */
    // 假设是3 1 1 3到 3 2 2 3，且此时已经处理了为了1 2 2 3
    for (int i = 0; i < output_dims[cur_idx]; i++) {
        broadcast_recursive(cur_idx + 1, last, input, output, cur2last_dim_mul,
                            input_offset, output_offset, output_dims,
                            input_dims, element_size);
    }
}

void wjy_broadcast(Tensor& output, Tensor& input) {
    // after unsequeeze, input.rank() == output.rank() 最后一个维度补1
    auto element_size = output.element_size();
    auto in_data = input.values<char>();
    auto out_data = output.values<char>();
    std::vector<int64_t> output_dims = output.dims();
    std::vector<int64_t> input_dims;
    int64_t output_offset = 0;
    int64_t input_offset = 0;
    auto out_strides = output.strides();
    std::vector<int64_t> stride_bits(output_dims.size());
    for (int64_t i = static_cast<int64_t>(output_dims.size()) - 1; i >= 0;
         --i) {
        stride_bits[i] = out_strides[i] * element_size;
    }
    int64_t last_not_eq_dim = static_cast<int64_t>(input.dims().size()) - 1;
    while (input.dim_size(last_not_eq_dim) ==
           output.dim_size(last_not_eq_dim)) {
        --last_not_eq_dim;
    }
    memcpy(out_data, in_data, stride_bits[last_not_eq_dim]);
    for (int64_t i = last_not_eq_dim; i >= 0; --i) {
        if (i == last_not_eq_dim) {
            Broadcast(out_data, stride_bits[i] * output_dims[i], output_dims[i],
                      stride_bits[i]);
            continue;
        }
        if (input.dim_size(i) == output.dim_size(i)) {
            for (int64_t j = 0; j < output.dim_size(j); ++j) {
                memcpy(out_data + output_offset, in_data + input_offset,
                       stride_bits[last_not_eq_dim]);
                Broadcast(
                    out_data + output_offset,
                    stride_bits[last_not_eq_dim] * output_dims[last_not_eq_dim],
                    output_dims[last_not_eq_dim], stride_bits[last_not_eq_dim]);
                output_offset += 0;
                input_offset += 0;
            }
        } else {
            // 这个可以合并，之后再优化
            Broadcast(out_data, stride_bits[i] * output_dims[i], output_dims[i],
                      stride_bits[i]);
        }
    }
}

}  // namespace kernels
}  // namespace wjy
