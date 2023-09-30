#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

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
void print_vec(const vector<T>& arr) {
    for (size_t i = 0; i < arr.size() - 1; i++) {
        cout << arr[i] << ", ";
    }
    cout << arr[arr.size() - 1] << endl;
}
int main() {
    vector<int> output(10, 0);
    output[0] = 3;
    print_vec<int>(output);
    Broadcast(output.data(), 1, 1, sizeof(int));
    print_vec<int>(output);
    return 0;
}