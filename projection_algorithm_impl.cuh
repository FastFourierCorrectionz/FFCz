#pragma once

// ProjectionSolver host function implementation
template <typename T>
ProjectionSolver<T>::ProjectionSolver(size_t Nx, size_t Ny, size_t Nz)
    : initialized_(false)
{
    params_.Nx = Nx;
    params_.Ny = Ny;
    params_.Nz = Nz;

    // Set up result array sizes
    results_.max_freq_size = Nx * Ny * (Nz / 2 + 1);
    results_.max_spatial_size = Nx * Ny * Nz;

    allocate_device_memory();
    setup_fft_plans();
}

template <typename T>
ProjectionSolver<T>::~ProjectionSolver()
{
    cleanup_fft_plans();
    deallocate_device_memory();
}

template <typename T>
void ProjectionSolver<T>::allocate_device_memory()
{
    size_t spatial_size = results_.max_spatial_size;
    size_t freq_size = results_.max_freq_size;

    // Core data arrays
    CHECK_CUDA(cudaMalloc(&params_.d_orig_data, spatial_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&params_.d_spatial_error, spatial_size * sizeof(T)));

    CHECK_CUDA(cudaMalloc(&params_.d_orig_freq, freq_size * sizeof(typename CufftTraits<T>::ComplexType)));
    CHECK_CUDA(cudaMalloc(&params_.d_freq_error, freq_size * sizeof(typename CufftTraits<T>::ComplexType)));

    // Full-size accumulator arrays
    CHECK_CUDA(cudaMalloc(&results_.d_freq_edits, 2 * freq_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&results_.d_spatial_edits, spatial_size * sizeof(T)));

    // Flag arrays
    CHECK_CUDA(cudaMalloc(&results_.d_freq_flag, 2 * freq_size * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&results_.d_spatial_flag, spatial_size * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&results_.d_freq_flag_pack, (2 * freq_size + 7) / 8 * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&results_.d_spatial_flag_pack, (spatial_size + 7) / 8 * sizeof(uint8_t)));

    // Compact edit arrays (pre-allocate maximum possible size)
    CHECK_CUDA(cudaMalloc(&results_.d_freq_edit_compact, 2 * freq_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&results_.d_spatial_edit_compact, spatial_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&results_.d_freq_edit_compact_quantized, 2 * freq_size * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&results_.d_spatial_edit_compact_quantized, spatial_size * sizeof(uint16_t)));

    // Temporary buffer for compaction
    CHECK_CUDA(cudaMalloc(&d_temp_write_pos_, sizeof(size_t)));

    // Initialize accumulators to zero
    reset_edit_accumulators();
}

template <typename T>
void ProjectionSolver<T>::deallocate_device_memory()
{
    // Core arrays
    if (params_.d_orig_data)
        cudaFree(params_.d_orig_data);
    if (params_.d_spatial_error)
        cudaFree(params_.d_spatial_error);
    if (params_.d_orig_freq)
        cudaFree(params_.d_orig_freq);
    if (params_.d_freq_error)
        cudaFree(params_.d_freq_error);
    if (params_.d_freq_indices)
        cudaFree(params_.d_freq_indices);
    if (params_.d_delta_k)
        cudaFree(params_.d_delta_k);

    // Accumulator arrays
    if (results_.d_freq_edits)
        cudaFree(results_.d_freq_edits);
    if (results_.d_spatial_edits)
        cudaFree(results_.d_spatial_edits);

    // Flag and compact arrays
    if (results_.d_freq_flag)
        cudaFree(results_.d_freq_flag);
    if (results_.d_spatial_flag)
        cudaFree(results_.d_spatial_flag);
    if (results_.d_freq_flag_pack)
        cudaFree(results_.d_freq_flag_pack);
    if (results_.d_spatial_flag_pack)
        cudaFree(results_.d_spatial_flag_pack);
    if (results_.d_freq_edit_compact)
        cudaFree(results_.d_freq_edit_compact);
    if (results_.d_spatial_edit_compact)
        cudaFree(results_.d_spatial_edit_compact);
    if (results_.d_freq_edit_compact_quantized)
        cudaFree(results_.d_freq_edit_compact_quantized);
    if (results_.d_spatial_edit_compact_quantized)
        cudaFree(results_.d_spatial_edit_compact_quantized);

    // Temporary arrays
    if (d_temp_write_pos_)
        cudaFree(d_temp_write_pos_);
}

template <typename T>
void ProjectionSolver<T>::setup_fft_plans()
{
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_forward, params_.Nx, params_.Ny, params_.Nz, CUFFT_R2C));
        CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_inverse, params_.Nx, params_.Ny, params_.Nz, CUFFT_C2R));
    }
    else
    {
        CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_forward, params_.Nx, params_.Ny, params_.Nz, CUFFT_D2Z));
        CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_inverse, params_.Nx, params_.Ny, params_.Nz, CUFFT_Z2D));
    }
}

template <typename T>
void ProjectionSolver<T>::cleanup_fft_plans()
{
    if (params_.fft_plan_forward)
        cufftDestroy(params_.fft_plan_forward);
    if (params_.fft_plan_inverse)
        cufftDestroy(params_.fft_plan_inverse);
}

template <typename T>
void ProjectionSolver<T>::reset_edit_accumulators()
{
    dim3 block(256);
    size_t max_size = std::max(2 * results_.max_freq_size, results_.max_spatial_size);
    dim3 grid((max_size + block.x - 1) / block.x);

    reset_accumulators<T><<<grid, block>>>(
        results_.d_freq_edits,
        results_.d_spatial_edits,
        results_.max_freq_size,
        results_.max_spatial_size);

    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_polytope()
{
    // Transform current spatial error to frequency domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftComplex *)params_.d_freq_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftDoubleComplex *)params_.d_freq_error));
    }

    // Project frequency components and accumulate edit distances
    dim3 block(256);
    dim3 grid((2 * params_.num_freq_constraints + block.x - 1) / block.x);

    project_frequency_constraints_direct<T><<<grid, block>>>(
        params_.d_freq_error,
        params_.d_freq_indices,
        params_.d_delta_k,
        results_.d_freq_edits,
        params_.Nx, params_.Ny, params_.Nz, results_.max_freq_size,
        params_.num_freq_constraints);

    CHECK_CUDA(cudaGetLastError());

    // Transform back to spatial domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse, (cufftComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse, (cufftDoubleComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }

    // Normalize after inverse FFT
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    T norm_factor = T(1) / static_cast<T>(spatial_size);

    dim3 spatial_block(256);
    dim3 spatial_grid((spatial_size + spatial_block.x - 1) / spatial_block.x);

    normalize_kernel<T><<<spatial_grid, spatial_block>>>(params_.d_spatial_error, norm_factor, spatial_size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_polytope_all_abs()
{
    // Transform current spatial error to frequency domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftComplex *)params_.d_freq_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftDoubleComplex *)params_.d_freq_error));
    }

    // Project frequency components and accumulate edit distances
    dim3 block(256);
    dim3 grid((2 * params_.num_freq_constraints + block.x - 1) / block.x);

    project_frequency_constraints_all_abs<T><<<grid, block>>>(
        params_.d_freq_error,
        params_.delta,
        results_.d_freq_edits,
        params_.Nx, params_.Ny, params_.Nz, results_.max_freq_size);

    CHECK_CUDA(cudaGetLastError());

    // Transform back to spatial domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse, (cufftComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse, (cufftDoubleComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }

    // Normalize after inverse FFT
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    T norm_factor = T(1) / static_cast<T>(spatial_size);

    dim3 spatial_block(256);
    dim3 spatial_grid((spatial_size + spatial_block.x - 1) / spatial_block.x);

    normalize_kernel<T><<<spatial_grid, spatial_block>>>(params_.d_spatial_error, norm_factor, spatial_size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_polytope_all_ptw()
{
    // Transform current spatial error to frequency domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftComplex *)params_.d_freq_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftDoubleComplex *)params_.d_freq_error));
    }

    // Project frequency components and accumulate edit distances
    dim3 block(256);
    dim3 grid((2 * params_.num_freq_constraints + block.x - 1) / block.x);

    project_frequency_constraints_all_ptw<T><<<grid, block>>>(
        params_.d_freq_error,
        params_.d_orig_freq,
        params_.delta_ptw,
        results_.d_freq_edits,
        params_.Nx, params_.Ny, params_.Nz, results_.max_freq_size);

    CHECK_CUDA(cudaGetLastError());

    // Transform back to spatial domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse, (cufftComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse, (cufftDoubleComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }

    // Normalize after inverse FFT
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    T norm_factor = T(1) / static_cast<T>(spatial_size);

    dim3 spatial_block(256);
    dim3 spatial_grid((spatial_size + spatial_block.x - 1) / spatial_block.x);

    normalize_kernel<T><<<spatial_grid, spatial_block>>>(params_.d_spatial_error, norm_factor, spatial_size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_polytope_partial_ptw()
{
    // Transform current spatial error to frequency domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftComplex *)params_.d_freq_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spatial_error,
                                 (cufftDoubleComplex *)params_.d_freq_error));
    }

    // Project frequency components and accumulate edit distances
    dim3 block(256);
    dim3 grid((2 * params_.num_freq_constraints + block.x - 1) / block.x);

    project_frequency_constraints_partial_ptw<T><<<grid, block>>>(
        params_.d_freq_error,
        params_.d_orig_freq,
        params_.d_freq_indices,
        params_.delta_ptw,
        results_.d_freq_edits,
        params_.Nx, params_.Ny, params_.Nz, results_.max_freq_size,
        params_.num_freq_constraints);

    CHECK_CUDA(cudaGetLastError());

    // Transform back to spatial domain
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse, (cufftComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }
    else
    {
        CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse, (cufftDoubleComplex *)params_.d_freq_error,
                                 params_.d_spatial_error));
    }

    // Normalize after inverse FFT
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    T norm_factor = T(1) / static_cast<T>(spatial_size);

    dim3 spatial_block(256);
    dim3 spatial_grid((spatial_size + spatial_block.x - 1) / spatial_block.x);

    normalize_kernel<T><<<spatial_grid, spatial_block>>>(params_.d_spatial_error, norm_factor, spatial_size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_spatial_box()
{
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    dim3 block(256);
    dim3 grid((spatial_size + block.x - 1) / block.x);

    project_spatial_constraints_direct<T><<<grid, block>>>(
        params_.d_spatial_error,
        params_.spatial_epsilon,
        results_.d_spatial_edits,
        spatial_size);

    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::extract_compact_representation()
{
    dim3 block(256);

    // Temporary arrays
    int *d_freq_int_flags = nullptr;
    int *d_spatial_int_flag = nullptr;
    CHECK_CUDA(cudaMalloc(&d_freq_int_flags, 2 * params_.num_freq_constraints * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_spatial_int_flag, results_.max_spatial_size * sizeof(int)));

    // Extract frequency flags
    dim3 freq_grid((2 * results_.max_freq_size + block.x - 1) / block.x);
    extract_active_flags<T><<<freq_grid, block>>>(
        results_.d_freq_edits,
        results_.d_freq_flag,
        d_freq_int_flags,
        2 * results_.max_freq_size,
        T(1e-7)); // Small threshold for numerical precision

    // Extract spatial flags
    dim3 spatial_grid((results_.max_spatial_size + block.x - 1) / block.x);
    extract_active_flags<T><<<spatial_grid, block>>>(
        results_.d_spatial_edits,
        results_.d_spatial_flag,
        d_spatial_int_flag,
        results_.max_spatial_size,
        T(1e-7));

    CHECK_CUDA(cudaGetLastError());

    // Compact frequency data
    compact_and_quantize_active_elements<T>(
        results_.d_freq_edits,
        results_.d_freq_flag,
        d_freq_int_flags,
        results_.d_freq_edit_compact,
        results_.d_freq_edit_compact_quantized,
        &results_.h_freq_min_edit,
        &results_.h_freq_max_edit,
        2 * results_.max_freq_size,
        &results_.h_num_active_freq);

    // Compact spatial data
    compact_and_quantize_active_elements<T>(
        results_.d_spatial_edits,
        results_.d_spatial_flag,
        d_spatial_int_flag,
        results_.d_spatial_edit_compact,
        results_.d_spatial_edit_compact_quantized,
        &results_.h_spatial_min_edit,
        &results_.h_spatial_max_edit,
        results_.max_spatial_size,
        &results_.h_num_active_spatial);

    CHECK_CUDA(cudaGetLastError());

    // Pack frequency flags
    size_t num_bytes;
    if (results_.h_num_active_freq > 0)
    {
        num_bytes = (2 * results_.max_freq_size + 7) / 8;
        dim3 freq_grid_pack((num_bytes + block.x - 1) / block.x);
        pack_bools<<<freq_grid, block>>>(
            results_.d_freq_flag,
            results_.d_freq_flag_pack,
            num_bytes);
    }

    // Pack spatial flags
    if (results_.h_num_active_spatial > 0)
    {
        num_bytes = (results_.max_spatial_size + 7) / 8;
        dim3 spatial_grid_pack((num_bytes + block.x - 1) / block.x);
        pack_bools<<<spatial_grid, block>>>(
            results_.d_spatial_flag,
            results_.d_spatial_flag_pack,
            num_bytes);
    }

    CHECK_CUDA(cudaGetLastError());

    // Cleanup temporary storage
    cudaFree(d_freq_int_flags);
    cudaFree(d_spatial_int_flag);
}

// Methods to retrieve results
template <typename T>
std::vector<bool> ProjectionSolver<T>::get_freq_flags()
{
    std::vector<bool> flags(2 * results_.max_freq_size);
    std::vector<char> temp_flags(2 * results_.max_freq_size);
    CHECK_CUDA(cudaMemcpy(temp_flags.data(), results_.d_freq_flag,
                          2 * results_.max_freq_size * sizeof(bool), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < 2 * results_.max_freq_size; ++i)
    {
        flags[i] = temp_flags[i];
    }
    return flags;
}

template <typename T>
std::vector<bool> ProjectionSolver<T>::get_spatial_flags()
{
    std::vector<bool> flags(results_.max_spatial_size);
    std::vector<char> temp_flags(results_.max_spatial_size);
    CHECK_CUDA(cudaMemcpy(temp_flags.data(), results_.d_spatial_flag,
                          results_.max_spatial_size * sizeof(bool), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < results_.max_spatial_size; ++i)
    {
        flags[i] = temp_flags[i];
    }
    return flags;
}

template <typename T>
std::vector<uint8_t> ProjectionSolver<T>::get_freq_flags_pack()
{
    size_t num_bytes = (2 * results_.max_freq_size + 7) / 8;
    std::vector<uint8_t> flags_pack(num_bytes);
    CHECK_CUDA(cudaMemcpy(flags_pack.data(), results_.d_freq_flag_pack,
                          num_bytes * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return flags_pack;
}

template <typename T>
std::vector<uint8_t> ProjectionSolver<T>::get_spatial_flags_pack()
{
    size_t num_bytes = (results_.max_spatial_size + 7) / 8;
    std::vector<uint8_t> flags_pack(num_bytes);
    CHECK_CUDA(cudaMemcpy(flags_pack.data(), results_.d_spatial_flag_pack,
                          num_bytes * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return flags_pack;
}

template <typename T>
std::vector<uint16_t> ProjectionSolver<T>::get_active_freq_quant_edit_distances()
{
    std::vector<uint16_t> distances(results_.h_num_active_freq);
    if (results_.h_num_active_freq > 0)
    {
        CHECK_CUDA(cudaMemcpy(distances.data(), results_.d_freq_edit_compact_quantized,
                              results_.h_num_active_freq * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }
    return distances;
}

template <typename T>
std::vector<uint16_t> ProjectionSolver<T>::get_active_spatial_quant_edit_distances()
{
    std::vector<uint16_t> distances(results_.h_num_active_spatial);
    if (results_.h_num_active_spatial > 0)
    {
        CHECK_CUDA(cudaMemcpy(distances.data(), results_.d_spatial_edit_compact_quantized,
                              results_.h_num_active_spatial * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }
    return distances;
}

template <typename T>
std::vector<T> ProjectionSolver<T>::get_active_freq_edit_distances()
{
    std::vector<T> distances(results_.h_num_active_freq);
    if (results_.h_num_active_freq > 0)
    {
        CHECK_CUDA(cudaMemcpy(distances.data(), results_.d_freq_edit_compact,
                              results_.h_num_active_freq * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return distances;
}

template <typename T>
std::vector<T> ProjectionSolver<T>::get_active_spatial_edit_distances()
{
    std::vector<T> distances(results_.h_num_active_spatial);
    if (results_.h_num_active_spatial > 0)
    {
        CHECK_CUDA(cudaMemcpy(distances.data(), results_.d_spatial_edit_compact,
                              results_.h_num_active_spatial * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return distances;
}

template <typename T>
std::vector<T> ProjectionSolver<T>::get_freq_edits_full()
{
    std::vector<T> edits(2 * results_.max_freq_size);
    CHECK_CUDA(cudaMemcpy(edits.data(), results_.d_freq_edits,
                          2 * results_.max_freq_size * sizeof(T), cudaMemcpyDeviceToHost));
    return edits;
}

template <typename T>
std::vector<T> ProjectionSolver<T>::get_spatial_edits_full()
{
    std::vector<T> edits(results_.max_spatial_size);
    CHECK_CUDA(cudaMemcpy(edits.data(), results_.d_spatial_edits,
                          results_.max_spatial_size * sizeof(T), cudaMemcpyDeviceToHost));
    return edits;
}

template <typename T>
T ProjectionSolver<T>::get_min_freq_edit()
{
    return results_.h_freq_min_edit;
}

template <typename T>
T ProjectionSolver<T>::get_max_freq_edit()
{
    return results_.h_freq_max_edit;
}

template <typename T>
T ProjectionSolver<T>::get_min_spatial_edit()
{
    return results_.h_spatial_min_edit;
}

template <typename T>
T ProjectionSolver<T>::get_max_spatial_edit()
{
    return results_.h_spatial_max_edit;
}

//
template <typename T>
void ProjectionSolver<T>::initialize(
    const T *h_orig_data,
    const size_t *h_freq_indices,
    const T *h_delta_k,
    T spatial_epsilon,
    size_t num_freq_constraints)
{
    params_.spatial_epsilon = spatial_epsilon;
    params_.num_freq_constraints = num_freq_constraints;

    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;

    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(params_.d_orig_data, h_orig_data,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    // Compute original frequency data
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftComplex *)params_.d_orig_freq));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftDoubleComplex *)params_.d_orig_freq));
    }

    // Allocate and copy frequency constraint data
    CHECK_CUDA(cudaMalloc(&params_.d_freq_indices, num_freq_constraints * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&params_.d_delta_k, num_freq_constraints * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(params_.d_freq_indices, h_freq_indices,
                          num_freq_constraints * sizeof(size_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(params_.d_delta_k, h_delta_k,
                          num_freq_constraints * sizeof(T), cudaMemcpyHostToDevice));

    initialized_ = true;
}

template <typename T>
void ProjectionSolver<T>::initialize_all_abs(
    const T *h_orig_data,
    T spatial_epsilon,
    T delta)
{
    params_.spatial_epsilon = spatial_epsilon;
    params_.delta = delta;
    params_.num_freq_constraints = params_.Nx * params_.Ny * (params_.Nz / 2 + 1);

    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;

    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(params_.d_orig_data, h_orig_data,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    // Compute original frequency data
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftComplex *)params_.d_orig_freq));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftDoubleComplex *)params_.d_orig_freq));
    }

    initialized_ = true;
}

template <typename T>
void ProjectionSolver<T>::initialize_all_ptw(
    const T *h_orig_data,
    T spatial_epsilon,
    T delta_ptw)
{
    params_.spatial_epsilon = spatial_epsilon;
    params_.delta_ptw = delta_ptw;
    params_.num_freq_constraints = params_.Nx * params_.Ny * (params_.Nz / 2 + 1);

    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;

    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(params_.d_orig_data, h_orig_data,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    // Compute original frequency data
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftComplex *)params_.d_orig_freq));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftDoubleComplex *)params_.d_orig_freq));
    }

    initialized_ = true;
}

template <typename T>
void ProjectionSolver<T>::initialize_partial_ptw(
    const T *h_orig_data,
    const size_t *h_freq_indices,
    const T delta_ptw,
    T spatial_epsilon,
    size_t num_freq_constraints)
{
    params_.spatial_epsilon = spatial_epsilon;
    params_.delta_ptw = delta_ptw;
    params_.num_freq_constraints = num_freq_constraints;

    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;

    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(params_.d_orig_data, h_orig_data,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    // Compute original frequency data
    if constexpr (std::is_same_v<T, float>)
    {
        CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftComplex *)params_.d_orig_freq));
    }
    else
    {
        CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_orig_data,
                                 (cufftDoubleComplex *)params_.d_orig_freq));
    }

    // Allocate and copy frequency constraint data
    CHECK_CUDA(cudaMalloc(&params_.d_freq_indices, num_freq_constraints * sizeof(size_t)));

    CHECK_CUDA(cudaMemcpy(params_.d_freq_indices, h_freq_indices,
                          num_freq_constraints * sizeof(size_t), cudaMemcpyHostToDevice));

    initialized_ = true;
}

// Method for checking only spatial domain convergence
template <typename T>
T ProjectionSolver<T>::compute_convergence()
{
    size_t spatial_size = results_.max_spatial_size;
    const int block_size = 256;
    int num_blocks = (spatial_size + block_size - 1) / block_size;

    T *d_block_results;
    CHECK_CUDA(cudaMalloc(&d_block_results, num_blocks * sizeof(T)));

    compute_convergence_metric<T><<<num_blocks, block_size>>>(
        params_.d_spatial_error,
        params_.spatial_epsilon,
        d_block_results,
        spatial_size);

    // Reduce on host for simplicity
    std::vector<T> h_block_results(num_blocks);
    CHECK_CUDA(cudaMemcpy(h_block_results.data(), d_block_results,
                          num_blocks * sizeof(T), cudaMemcpyDeviceToHost));

    T max_violation = 0;
    for (int i = 0; i < num_blocks; ++i)
    {
        max_violation = std::max(max_violation, h_block_results[i]);
    }

    cudaFree(d_block_results);
    return max_violation;
}

template <typename T>
void ProjectionSolver<T>::solve(
    T *h_spatial_error,
    int max_iterations,
    T convergence_tolerance)
{
    if (!initialized_)
    {
        throw std::runtime_error("Solver not initialized");
    }

    size_t spatial_size = results_.max_spatial_size;

    // Copy initial spatial error
    CHECK_CUDA(cudaMemcpy(params_.d_spatial_error, h_spatial_error,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    results_.iteration_count = 0;

    for (int iter = 0; iter < max_iterations; iter++)
    {
        // Project onto frequency polytope (accumulates in freq_edits)
        project_onto_frequency_polytope();

        // Check spatial convergence
        T convergence = compute_convergence();
        results_.convergence_metric = convergence;
        results_.iteration_count++;

        if (convergence < convergence_tolerance)
        {
            break;
        }

        // Project onto spatial box (accumulates in spatial_edits)
        project_onto_spatial_box();
    }

    printf("Total number of iterations: %d\n", results_.iteration_count);

    // Extract compact representation from accumulators
    extract_compact_representation();
}

template <typename T>
void ProjectionSolver<T>::solve_all_abs(
    T *h_spatial_error,
    int max_iterations,
    T convergence_tolerance)
{
    if (!initialized_)
    {
        throw std::runtime_error("Solver not initialized");
    }

    size_t spatial_size = results_.max_spatial_size;

    // Copy initial spatial error
    CHECK_CUDA(cudaMemcpy(params_.d_spatial_error, h_spatial_error,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    results_.iteration_count = 0;

    for (int iter = 0; iter < max_iterations; iter++)
    {
        // Project onto frequency polytope (accumulates in freq_edits)
        project_onto_frequency_polytope_all_abs();

        // Check spatial convergence
        T convergence = compute_convergence();
        results_.convergence_metric = convergence;
        results_.iteration_count++;

        if (convergence < convergence_tolerance)
        {
            break;
        }

        // Project onto spatial box (accumulates in spatial_edits)
        project_onto_spatial_box();
    }

    printf("Total number of iterations: %d\n", results_.iteration_count);

    // Extract compact representation from accumulators
    extract_compact_representation();
}

template <typename T>
void ProjectionSolver<T>::solve_all_ptw(
    T *h_spatial_error,
    int max_iterations,
    T convergence_tolerance)
{
    if (!initialized_)
    {
        throw std::runtime_error("Solver not initialized");
    }

    size_t spatial_size = results_.max_spatial_size;

    // Copy initial spatial error
    CHECK_CUDA(cudaMemcpy(params_.d_spatial_error, h_spatial_error,
                          spatial_size * sizeof(T), cudaMemcpyHostToDevice));

    results_.iteration_count = 0;

    for (int iter = 0; iter < max_iterations; iter++)
    {
        // Project onto frequency polytope (accumulates in freq_edits)
        project_onto_frequency_polytope_all_ptw();

        // Check spatial convergence
        T convergence = compute_convergence();
        results_.convergence_metric = convergence;
        results_.iteration_count++;

        if (convergence < convergence_tolerance)
        {
            break;
        }

        // Project onto spatial box (accumulates in spatial_edits)
        project_onto_spatial_box();
    }

    printf("Total number of iterations: %d\n", results_.iteration_count);

    // Extract compact representation from accumulators
    extract_compact_representation();
}

template <typename T>
void ProjectionSolver<T>::get_results(T *h_spatial_error)
{
    size_t spatial_size = params_.Nx * params_.Ny * params_.Nz;
    CHECK_CUDA(cudaMemcpy(h_spatial_error, params_.d_spatial_error,
                          spatial_size * sizeof(T), cudaMemcpyDeviceToHost));
}

// template <typename T>
// void ProjectionSolver<T>::get_results(T *h_spatial_error)
// {
//     dim3 block(256);

//     dim3 freq_grid((2 * results_.max_freq_size + block.x - 1) / block.x);
//     T *dequant_freq_edits_split;
//     CHECK_CUDA(cudaMalloc(&dequant_freq_edits_split, 2 * results_.max_freq_size * sizeof(T)));

//     quantize_and_dequantize<T><<<freq_grid, block>>>(
//         results_.d_freq_edits,
//         dequant_freq_edits_split,
//         results_.h_freq_min_edit,
//         results_.h_freq_max_edit,
//         2 * results_.max_freq_size,
//         T(1e-7));

//     dim3 freq_grid2((results_.max_freq_size + block.x - 1) / block.x);
//     typename CufftTraits<T>::ComplexType *dequant_freq_edits;
//     CHECK_CUDA(cudaMalloc(&dequant_freq_edits, results_.max_freq_size * sizeof(typename CufftTraits<T>::ComplexType)));
//     T *d_spatial_error;
//     CHECK_CUDA(cudaMalloc(&d_spatial_error, results_.max_spatial_size * sizeof(T)));

//     frequency_float_to_complex<T><<<freq_grid2, block>>>(
//         results_.d_freq_edits,
//         dequant_freq_edits,
//         results_.max_freq_size);

//     // IFFT
//     if constexpr (std::is_same_v<T, float>)
//     {
//         CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse, (cufftComplex *)dequant_freq_edits,
//                                  d_spatial_error));
//     }
//     else
//     {
//         CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse, (cufftDoubleComplex *)dequant_freq_edits,
//                                  d_spatial_error));
//     }
//     T norm_factor = T(1) / static_cast<T>(results_.max_spatial_size);

//     dim3 spatial_grid((results_.max_spatial_size + block.x - 1) / block.x);

//     normalize_kernel<T><<<spatial_grid, block>>>(params_.d_spatial_error, norm_factor, results_.max_spatial_size);
//     CHECK_CUDA(cudaGetLastError());

//     T *dequant_spatial_edits;
//     CHECK_CUDA(cudaMalloc(&dequant_spatial_edits, results_.max_spatial_size * sizeof(T)));

//     quantize_and_dequantize<T><<<spatial_grid, block>>>(
//         results_.d_spatial_edits,
//         dequant_spatial_edits,
//         results_.h_spatial_min_edit,
//         results_.h_spatial_max_edit,
//         results_.max_spatial_size,
//         T(1e-7));

//     add_two_arrays<T><<<spatial_grid, block>>>(
//         dequant_spatial_edits,
//         d_spatial_error,
//         results_.max_spatial_size);

//     add_two_arrays<T><<<spatial_grid, block>>>(
//         params_.d_spatial_error,
//         d_spatial_error,
//         results_.max_spatial_size);

//     CHECK_CUDA(cudaMemcpy(h_spatial_error, d_spatial_error, results_.max_spatial_size * sizeof(T), cudaMemcpyDeviceToHost));

//     cudaFree(dequant_freq_edits_split);
//     cudaFree(dequant_freq_edits);
//     cudaFree(d_spatial_error);
//     cudaFree(dequant_spatial_edits);
// }