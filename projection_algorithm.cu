
#include "projection_algorithm.cuh"

// Template kernel definitions here

// Device helper functions
template <typename T>
__device__ __forceinline__ T complex_magnitude_squared(const typename CufftTraits<T>::ComplexType &c);

template <>
__device__ __forceinline__ float complex_magnitude_squared<float>(const cufftComplex &c)
{
    return c.x * c.x + c.y * c.y;
}

template <>
__device__ __forceinline__ double complex_magnitude_squared<double>(const cufftDoubleComplex &c)
{
    return c.x * c.x + c.y * c.y;
}

template <typename T>
__device__ __forceinline__ void complex_scale(typename CufftTraits<T>::ComplexType &c, T scale);

template <>
__device__ __forceinline__ void complex_scale<float>(cufftComplex &c, float scale)
{
    c.x *= scale;
    c.y *= scale;
}

template <>
__device__ __forceinline__ void complex_scale<double>(cufftDoubleComplex &c, double scale)
{
    c.x *= scale;
    c.y *= scale;
}

template <typename T>
__device__ __forceinline__ T abs_val(T x);

template <>
__device__ __forceinline__ float abs_val(float x)
{
    return fabsf(x);
}

template <>
__device__ __forceinline__ double abs_val(double x)
{
    return fabs(x);
}

template <typename T>
__device__ __forceinline__ T sqrt_val(T x);

template <>
__device__ __forceinline__ float sqrt_val(float x)
{
    return sqrtf(x);
}

template <>
__device__ __forceinline__ double sqrt_val(double x)
{
    return sqrt(x);
}

template <typename T>
__device__ __forceinline__ T log10_val(T x);

template <>
__device__ __forceinline__ float log10_val(float x)
{
    return log10f(x);
}

template <>
__device__ __forceinline__ double log10_val(double x)
{
    return log10(x);
}

// Normalization kernel
template <typename T>
__global__ void normalize_kernel(T *data, T factor, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] *= factor;
}

// Frequency projection kernel
template <typename T>
__global__ void project_frequency_constraints_direct(
    typename CufftTraits<T>::ComplexType *curr_freq_err, // This is freq_error from FFT(spatial_error)
    const size_t *freq_constraint_indices,
    const T *delta_k,
    T *freq_edits, // Accumulator: edits from orig_freq
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size,
    size_t num_constraints)
{
    size_t constraint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (constraint_id >= 2 * num_constraints)
        return;

    size_t idx = constraint_id % num_constraints;
    size_t is_img_part = constraint_id / num_constraints;
    size_t freq_idx = freq_constraint_indices[idx];
    T delta = delta_k[idx];

    T curr_err, edit_dist;
    if (is_img_part > 0)
    {
        curr_err = curr_freq_err[freq_idx].y;
        if (curr_err > delta)
        {
            edit_dist = delta - curr_err;
            atomicAdd(&freq_edits[freq_idx + max_freq_size], edit_dist); // Accumulate frequency edit
            curr_freq_err[freq_idx].y = delta;
        }
        else if (curr_err < -delta)
        {
            edit_dist = -delta - curr_err;
            atomicAdd(&freq_edits[freq_idx + max_freq_size], edit_dist); // Accumulate frequency edit
            curr_freq_err[freq_idx].y = -delta;
        }
    }
    else
    {
        curr_err = curr_freq_err[freq_idx].x;
        if (curr_err > delta)
        {
            edit_dist = delta - curr_err;
            atomicAdd(&freq_edits[freq_idx], edit_dist); // Accumulate frequency edit
            curr_freq_err[freq_idx].x = delta;
        }
        else if (curr_err < -delta)
        {
            edit_dist = -delta - curr_err;
            atomicAdd(&freq_edits[freq_idx], edit_dist); // Accumulate frequency edit
            curr_freq_err[freq_idx].x = -delta;
        }
    }
}

// Frequency projection kernel (ALL frequency components with ABSOLUTE bound)
template <typename T>
__global__ void project_frequency_constraints_all_abs(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const T delta,
    T *freq_edits,
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_freq_size)
        return;

    // Process real part
    T curr_err = curr_freq_err[idx].x;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].x = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].x = -delta;
    }

    // Process imaginary part (no need to skip DC and Nyquist components)
    curr_err = curr_freq_err[idx].y;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].y = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].y = -delta;
    }
}

// Frequency projection kernel (ALL frequency components with POINTWISE relative bound)
template <typename T>
__global__ void project_frequency_constraints_all_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const T delta_ptw,
    T *freq_edits,
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_freq_size)
        return;

    // separated condition for DC component
    if (idx == 0)
    {
        T lower_bound = orig_freq[0].x * (1 / sqrt_val(1 + delta_ptw) - 1);
        T upper_bound = orig_freq[0].x * (1 / sqrt_val(1 - delta_ptw) - 1);
        T curr_err = curr_freq_err[0].x;
        if (curr_err > upper_bound || curr_err < lower_bound)
        {
            T edit_dist = -curr_err;
            atomicAdd(&freq_edits[idx], edit_dist); // edit DC component error close to 0
            curr_freq_err[0].x = 0;
        }
        return;
    }

    // conditions for non-DC components
    T orig_mag = complex_magnitude_squared<T>(orig_freq[idx]);
    typename CufftTraits<T>::ComplexType curr_freq;
    curr_freq.x = orig_freq[idx].x + curr_freq_err[idx].x;
    curr_freq.y = orig_freq[idx].y + curr_freq_err[idx].y;
    T curr_mag = complex_magnitude_squared<T>(curr_freq);
    T curr_ratio = curr_mag / orig_mag;
    if (curr_ratio <= sqrt_val(1 + delta_ptw) && curr_ratio >= sqrt_val(1 - delta_ptw))
        return;

    T delta = fmin(1 - sqrt_val(1 - delta_ptw), sqrt_val(1 + delta_ptw) - 1) / sqrt_val(2.0f);

    // Process real part
    T curr_err = curr_freq_err[idx].x;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].x = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].x = -delta;
    }

    // Process imaginary part (no need to skip DC and Nyquist components)
    curr_err = curr_freq_err[idx].y;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].y = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[idx].y = -delta;
    }
}

// Frequency projection kernel (PARTIAL frequency components with POINTWISE relative bound)
template <typename T>
__global__ void project_frequency_constraints_partial_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const size_t *freq_constraint_indices,
    const T delta_ptw,
    T *freq_edits,
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size,
    size_t num_constraints)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_constraints)
        return;

    // separated condition for DC component
    if (idx == 0)
    {
        T lower_bound = orig_freq[0].x * (1 / sqrt_val(1 + delta_ptw) - 1);
        T upper_bound = orig_freq[0].x * (1 / sqrt_val(1 - delta_ptw) - 1);
        T curr_err = curr_freq_err[0].x;
        if (curr_err > upper_bound || curr_err < lower_bound)
        {
            T edit_dist = -curr_err;
            atomicAdd(&freq_edits[idx], edit_dist);
            curr_freq_err[0].x = 0;
        }
        return;
    }

    // conditions for non-DC components
    size_t freq_idx = freq_constraint_indices[idx];
    T orig_mag = complex_magnitude_squared<T>(orig_freq[freq_idx]);
    typename CufftTraits<T>::ComplexType curr_freq;
    curr_freq.x = orig_freq[freq_idx].x + curr_freq_err[freq_idx].x;
    curr_freq.y = orig_freq[freq_idx].y + curr_freq_err[freq_idx].y;
    T curr_mag = complex_magnitude_squared<T>(curr_freq);
    T curr_ratio = curr_mag / orig_mag;
    if (curr_ratio <= sqrt_val(1 + delta_ptw) && curr_ratio >= sqrt_val(1 - delta_ptw))
        return;

    T delta = fmin(1 - sqrt_val(1 - delta_ptw), sqrt_val(1 + delta_ptw) - 1) / sqrt_val(2.0f);

    // Process real part
    T curr_err = curr_freq_err[freq_idx].x;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[freq_idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[freq_idx].x = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[freq_idx], edit_dist); // Accumulate frequency edit
        curr_freq_err[freq_idx].x = -delta;
    }

    // Process imaginary part (no need to skip DC and Nyquist components)
    curr_err = curr_freq_err[freq_idx].y;
    if (curr_err > delta)
    {
        T edit_dist = delta - curr_err;
        atomicAdd(&freq_edits[freq_idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[freq_idx].y = delta;
    }
    else if (curr_err < -delta)
    {
        T edit_dist = -delta - curr_err;
        atomicAdd(&freq_edits[freq_idx + max_freq_size], edit_dist); // Accumulate frequency edit
        curr_freq_err[freq_idx].y = -delta;
    }
}

// Spatial projection kernel
template <typename T>
__global__ void project_spatial_constraints_direct(
    T *spatial_error,
    T spatial_epsilon,
    T *spatial_edits,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    T curr_err = spatial_error[idx];

    T edit_distance = 0;

    if (curr_err > spatial_epsilon)
    {
        spatial_edits[idx] += spatial_epsilon - curr_err; // Accumulate total spatial edit distance
        spatial_error[idx] = spatial_epsilon;             // Update spatial error
    }
    else if (curr_err < -spatial_epsilon)
    {
        spatial_edits[idx] += -spatial_epsilon - curr_err; // Accumulate total spatial edit distance
        spatial_error[idx] = -spatial_epsilon;             // Update spatial error
    }
}

// Utility kernels
// Extract flags from accumulator arrays
template <typename T>
__global__ void extract_active_flags(
    const T *edit_array,
    bool *flag_array,
    int *int_flag_array,
    size_t total_size,
    T threshold)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    flag_array[idx] = (abs_val(edit_array[idx]) > threshold);
    int_flag_array[idx] = flag_array[idx] ? 1 : 0;
}

// Pack boolean vector into 8-bit elements
__global__ void pack_bools(
    const bool *__restrict__ flag,
    uint8_t *__restrict__ flag_pack,
    size_t num_bytes)
{
    size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = byte_idx * 8;

    if (start >= num_bytes)
        return;

    uint8_t packed = 0;

#pragma unroll
    for (size_t i = 0; i < 8; ++i)
    {
        size_t idx = start + i;
        if (idx < num_bytes && flag[idx])
        {
            packed |= (1u << i);
        }
    }
    flag_pack[byte_idx] = packed;
}

// Compact using precomputed prefix sum positions
template <typename T>
__global__ void compact_with_prefix_positions(
    const T *edit_array,
    const bool *flag_array,
    const int *prefix_positions, // Exclusive prefix sum of flags
    T *compact_values,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    if (flag_array[idx])
    {
        int write_pos = prefix_positions[idx]; // Position from prefix sum
        compact_values[write_pos] = edit_array[idx];
    }
}

template <typename T>
__global__ void compact_and_quantize_with_prefix_positions(
    const T *edit_array,
    const bool *flag_array,
    const int *prefix_positions, // Exclusive prefix sum of flags
    T *compact_values,
    T *min_edit,
    T *max_edit,
    uint16_t *quant_values,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    if (flag_array[idx])
    {
        int write_pos = prefix_positions[idx]; // Position from prefix sum
        compact_values[write_pos] = edit_array[idx];
        // Quantization
        T normalized_edit = (edit_array[idx] - *min_edit) / (*max_edit - *min_edit);
        if (normalized_edit > 0.5)
        {
            quant_values[write_pos] = ceil(normalized_edit * ((1u << 16) - 1));
        }
        else
        {
            quant_values[write_pos] = floor(normalized_edit * ((1u << 16) - 1));
        }
    }
}

// Compact active elements
template <typename T>
void compact_and_quantize_active_elements(
    const T *edit_array,
    const bool *flag_array,
    const int *int_flag_array,
    T *compact_values,
    uint16_t *quant_values,
    T *h_min_edit,
    T *h_max_edit,
    size_t total_size,
    size_t *h_num_active)
{
    dim3 block_size(256);
    dim3 grid_size((total_size + block_size.x - 1) / block_size.x);

    // Find max and min edits for quantization later
    T *d_min_edit, *d_max_edit;
    CHECK_CUDA(cudaMalloc(&d_min_edit, sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_max_edit, sizeof(T)));

    void *d_temp_storage2 = nullptr;
    size_t temp_storage_bytes2 = 0;
    cub::DeviceReduce::Min(d_temp_storage2, temp_storage_bytes2, edit_array, d_min_edit, total_size);
    CHECK_CUDA(cudaMalloc(&d_temp_storage2, temp_storage_bytes2));
    cub::DeviceReduce::Min(d_temp_storage2, temp_storage_bytes2, edit_array, d_min_edit, total_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    cub::DeviceReduce::Max(d_temp_storage2, temp_storage_bytes2, edit_array, d_max_edit, total_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_min_edit, d_min_edit, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_max_edit, d_max_edit, sizeof(T), cudaMemcpyDeviceToHost));

    // Temporary arrays
    cudaStream_t stream = 0;
    int *d_prefix_positions = nullptr;
    void *d_temp_storage = nullptr;
    CHECK_CUDA(cudaMalloc(&d_prefix_positions, total_size * sizeof(int)));

    // Compute exclusive prefix sum using CUB
    size_t temp_storage_bytes = 0;

    // First call to get required storage size
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        int_flag_array, d_prefix_positions, total_size, stream);

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Second call to compute prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        int_flag_array, d_prefix_positions, total_size, stream);

    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error != cudaSuccess)
        throw cuda_error;

    // Count active elements
    int h_last_flag, h_last_prefix;
    CHECK_CUDA(cudaMemcpyAsync(&h_last_flag, &int_flag_array[total_size - 1], sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpyAsync(&h_last_prefix, &d_prefix_positions[total_size - 1], sizeof(int), cudaMemcpyDeviceToHost));

    int h_total_active = h_last_prefix + h_last_flag;
    *h_num_active = static_cast<size_t>(h_total_active);

    // Compact elements using computed positions
    if (h_total_active > 0)
    {
        compact_and_quantize_with_prefix_positions<T><<<grid_size, block_size>>>(
            edit_array,
            flag_array,
            d_prefix_positions,
            compact_values,
            d_min_edit,
            d_max_edit,
            quant_values,
            total_size);

        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemcpy(&h_min_edit, d_min_edit, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_max_edit, d_max_edit, sizeof(T), cudaMemcpyDeviceToHost));

    // Cleanup temporary storage
    cudaFree(d_prefix_positions);
    cudaFree(d_temp_storage);
    cudaFree(d_temp_storage2);
    cudaFree(d_min_edit);
    cudaFree(d_max_edit);
}

// Reset accumulator arrays
template <typename T>
__global__ void reset_accumulators(
    T *freq_edits,
    T *spatial_edits,
    size_t freq_size,
    size_t spatial_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x; // For better memory coalescing

    for (size_t i = idx; i < 2 * freq_size; i += stride)
    {
        freq_edits[i] = T(0);
    }

    for (size_t i = idx; i < spatial_size; i += stride)
    {
        spatial_edits[i] = T(0);
    }
}

// Convergence metric kernel
template <typename T>
__global__ void compute_convergence_metric(
    const T *spatial_error,
    T spatial_epsilon,
    T *convergence_result,
    size_t total_size)
{
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T local_violation = 0.0;
    if (idx < total_size)
    {
        T error = spatial_error[idx];
        if (abs_val(error) > spatial_epsilon)
        {
            local_violation = abs_val(error) - spatial_epsilon;
        }
    }

    sdata[tid] = local_violation;
    __syncthreads();

    // Reduction to find maximum violation
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        convergence_result[blockIdx.x] = sdata[0];
    }
}

// Quantize and dequantize
template <typename T>
__global__ void quantize_and_dequantize(
    const T *org_arr,
    T *dequant_arr,
    T min_val,
    T max_val,
    size_t total_size,
    T threshold)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    T org_val = org_arr[idx];
    if (abs_val(org_val) > threshold)
    {
        T normalized_edit = (org_val - min_val) / (max_val - min_val);
        T quant_val;
        if (normalized_edit > 0.5)
            quant_val = ceil(normalized_edit * ((1u << 16) - 1));
        else
            quant_val = floor(normalized_edit * ((1u << 16) - 1));
        dequant_arr[idx] = quant_val / ((1u << 16) - 1) * (max_val - min_val) + min_val;
    }
    else
    {
        dequant_arr[idx] = 0;
    }
}

// Transform split frequency to complex frequency
template <typename T>
__global__ void frequency_float_to_complex(
    const T *freq_split,
    typename CufftTraits<T>::ComplexType *freq_complex,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    freq_complex[idx].x = freq_split[idx];
    freq_complex[idx].y = freq_split[idx + total_size];
}

// Add two arrays
template <typename T>
__global__ void add_two_arrays(
    const T *arr1,
    T *arr2,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    arr2[idx] += arr1[idx];
}

// Kernel instantiations
template __global__ void normalize_kernel<float>(float *, float, size_t);
template __global__ void normalize_kernel<double>(double *, double, size_t);

template __global__ void project_frequency_constraints_direct<float>(
    cufftComplex *, const size_t *, const float *, float *, size_t, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_direct<double>(
    cufftDoubleComplex *, const size_t *, const double *, double *, size_t, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_all_abs<float>(
    cufftComplex *, const float, float *, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_all_abs<double>(
    cufftDoubleComplex *, const double, double *, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_all_ptw<float>(
    cufftComplex *, const cufftComplex *, const float, float *, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_all_ptw<double>(
    cufftDoubleComplex *, const cufftDoubleComplex *, const double, double *, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_partial_ptw<float>(
    cufftComplex *, const cufftComplex *, const size_t *, const float, float *, size_t, size_t, size_t, size_t, size_t);

template __global__ void project_frequency_constraints_partial_ptw<double>(
    cufftDoubleComplex *, const cufftDoubleComplex *, const size_t *, const double, double *, size_t, size_t, size_t, size_t, size_t);

template __global__ void project_spatial_constraints_direct<float>(
    float *, float, float *, size_t);

template __global__ void project_spatial_constraints_direct<double>(
    double *, double, double *, size_t);

template __global__ void extract_active_flags<float>(
    const float *, bool *, int *, size_t, float);

template __global__ void extract_active_flags<double>(
    const double *, bool *, int *, size_t, double);

template __global__ void compact_with_prefix_positions<float>(
    const float *, const bool *, const int *, float *, size_t);

template __global__ void compact_with_prefix_positions<double>(
    const double *, const bool *, const int *, double *, size_t);

template __global__ void compact_and_quantize_with_prefix_positions<float>(
    const float *, const bool *, const int *, float *, float *, float *, uint16_t *, size_t);

template __global__ void compact_and_quantize_with_prefix_positions<double>(
    const double *, const bool *, const int *, double *, double *, double *, uint16_t *, size_t);

template void compact_and_quantize_active_elements<float>(
    const float *, const bool *, const int *, float *, uint16_t *, float *, float *, size_t, size_t *);

template void compact_and_quantize_active_elements<double>(
    const double *, const bool *, const int *, double *, uint16_t *, double *, double *, size_t, size_t *);

template __global__ void reset_accumulators<float>(
    float *, float *, size_t, size_t);

template __global__ void reset_accumulators<double>(
    double *, double *, size_t, size_t);

template __global__ void compute_convergence_metric<float>(
    const float *, float, float *, size_t);

template __global__ void compute_convergence_metric<double>(
    const double *, double, double *, size_t);

template __global__ void quantize_and_dequantize<float>(
    const float *, float *, float, float, size_t, float);

template __global__ void quantize_and_dequantize<double>(
    const double *, double *, double, double, size_t, double);

template __global__ void frequency_float_to_complex<float>(
    const float *, cufftComplex *, size_t);

template __global__ void frequency_float_to_complex<double>(
    const double *, cufftDoubleComplex *, size_t);

template __global__ void add_two_arrays<float>(
    const float *, float *, size_t);

template __global__ void add_two_arrays<double>(
    const double *, double *, size_t);