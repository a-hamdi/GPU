#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>


extern "C" void solution(const float* input, float* output, size_t N) {
    thrust::device_ptr<const float> dev_input(input);
    thrust::device_ptr<float> dev_output(output);
    thrust::inclusive_scan(dev_input, dev_input + N, dev_output, thrust::multiplies<float>());
}
