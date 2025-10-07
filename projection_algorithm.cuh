#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <vector>
#include <iostream>
#include <cmath>

// Error checking macros
#define CHECK_CUDA(call)                                                                                                     \
    do                                                                                                                       \
    {                                                                                                                        \
        cudaError_t error = call;                                                                                            \
        if (error != cudaSuccess)                                                                                            \
        {                                                                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                                                         \
        }                                                                                                                    \
    } while (0)

#define CHECK_CUFFT(call)                                                                                 \
    do                                                                                                    \
    {                                                                                                     \
        cufftResult error = call;                                                                         \
        if (error != CUFFT_SUCCESS)                                                                       \
        {                                                                                                 \
            std::cerr << "CUFFT error at " << __FILE__ << ":" << __LINE__ << " - " << error << std::endl; \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

// Template trait for CUFFT types
template <typename T>
struct CufftTraits
{
};

template <>
struct CufftTraits<float>
{
    using ComplexType = cufftComplex;
    static const cufftType R2C = CUFFT_R2C;
    static const cufftType C2R = CUFFT_C2R;
};

template <>
struct CufftTraits<double>
{
    using ComplexType = cufftDoubleComplex;
    static const cufftType R2C = CUFFT_D2Z;
    static const cufftType C2R = CUFFT_Z2D;
};

// Structure to hold projection parameters
template <typename T>
struct ProjectionParams
{
    T *d_orig_data;                                     // Original data (Nx*Ny*Nz) - FIXED REFERENCE
    T *d_spatial_error;                                 // Current spatial error vector (Nx*Ny*Nz)
    typename CufftTraits<T>::ComplexType *d_orig_freq;  // Original frequency data - FIXED REFERENCE
    typename CufftTraits<T>::ComplexType *d_freq_error; // Current frequency errors (Nx*Ny*(Nz/2+1))
    size_t *d_freq_indices;                             // Indices of frequency components to preserve
    T *d_delta_k;                                       // Tolerance for each frequency component
    T delta;                                            // Absolute tolerance for all frequency components
    T delta_ptw;                                        // Pointwise tolerance for all frequency components
    T spatial_epsilon;                                  // Spatial constraint tolerance
    size_t Nx, Ny, Nz;                                  // Grid dimensions
    size_t num_freq_constraints;                        // Number of frequency constraints
    cufftHandle fft_plan_forward;                       // FFT plan (spatial to frequency)
    cufftHandle fft_plan_inverse;                       // IFFT plan (frequency to spatial)
};

// Simplified result structure - only accumulators and references
template <typename T>
struct IterationResults
{
    // Accumulator arrays (the core storage)
    T *d_freq_edits;    // Size: Nx*Ny*(Nz/2+1)*2 - accumulated edit distances from orig_freq
    T *d_spatial_edits; // Size: Nx*Ny*Nz - accumulated edit distances from orig_spatial (zero)

    // Compact representation (extracted when needed)
    bool *d_freq_flag;            // Size: Nx*Ny*(Nz/2+1)*2 - which components are active
    bool *d_spatial_flag;         // Size: Nx*Ny*Nz - which components are active
    uint8_t *d_freq_flag_pack;    // Size: Nx*Ny*(Nz/2+1)*2/8
    uint8_t *d_spatial_flag_pack; // Size: Nx*Ny*Nz/8
    T *d_freq_edit_compact;       // Variable size - only non-zero edit distances
    T *d_spatial_edit_compact;    // Variable size - only non-zero edit distances

    // Compact and quantized representation
    uint16_t *d_freq_edit_compact_quantized;    // Variable size - only non-zero quantized edit distances
    uint16_t *d_spatial_edit_compact_quantized; // Variable size - only non-zero quantized edit distances
    T h_freq_min_edit;
    T h_spatial_min_edit;
    T h_freq_max_edit;
    T h_spatial_max_edit;

    // Counters
    size_t h_num_active_freq;    // Host copy of active frequency count
    size_t h_num_active_spatial; // Host copy of active spatial count

    T convergence_metric; // Overall convergence measure
    int iteration_count;  // Current iteration number

    // Array sizes
    size_t max_freq_size;    // Nx*Ny*(Nz/2+1)
    size_t max_spatial_size; // Nx*Ny*Nz
};

// Kernel declarations
template <typename T>
__global__ void normalize_kernel(T *data, T factor, size_t size);

template <typename T>
__global__ void project_frequency_constraints_direct(
    typename CufftTraits<T>::ComplexType *curr_freq_err, // temp_freq from FFT(spatial_error)
    const size_t *freq_constraint_indices,
    const T *delta_k,
    T *freq_edits, // Accumulator array - edits from orig_freq
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size,
    size_t num_constraints);

// preserve ALL frequency components with a uniform ABSolute error bound
template <typename T>
__global__ void project_frequency_constraints_all_abs(
    typename CufftTraits<T>::ComplexType *curr_freq_err, // temp_freq from FFT(spatial_error)
    const T delta,
    T *freq_edits, // Accumulator array - edits from orig_freq
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size);

// for power spectrum: preserve ALL frequency components with a POINTWISE relative error bound
template <typename T>
__global__ void project_frequency_constraints_all_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const T delta_ptw,
    T *freq_edits, // Accumulator array - edits from orig_freq
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size);

// for power spectrum: preserve PARTIAL frequency components with a POINTWISE relative error bound
template <typename T>
__global__ void project_frequency_constraints_partial_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const size_t *freq_constraint_indices,
    const T delta_ptw,
    T *freq_edits, // Accumulator array - edits from orig_freq
    size_t Nx, size_t Ny, size_t Nz, size_t max_freq_size,
    size_t num_constraints);

template <typename T>
__global__ void project_spatial_constraints_direct(
    T *spatial_error,
    T spatial_epsilon,
    T *spatial_edits, // Accumulator array - edits from zero (original is perfect)
    size_t total_size);

template <typename T>
__global__ void extract_active_flags(
    const T *edit_array,
    bool *flag_array,
    int *int_flag_array,
    size_t total_size,
    T threshold = T(0));

__global__ void pack_bools(
    const bool *__restrict__ flag,
    uint8_t *__restrict__ flag_pack,
    size_t num_bytes);

template <typename T>
__global__ void compact_with_prefix_positions(
    const T *edit_array,
    const bool *flag_array,
    const int *prefix_positions,
    T *compact_values,
    size_t total_size);

template <typename T>
__global__ void compact_and_quantize_with_prefix_positions(
    const T *edit_array,
    const bool *flag_array,
    const int *prefix_positions,
    T *compact_values,
    T *min_edit,
    T *max_edit,
    uint16_t *quant_values,
    size_t total_size);

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
    size_t *h_num_active);

template <typename T>
__global__ void reset_accumulators(
    T *freq_edits,
    T *spatial_edits,
    size_t freq_size,
    size_t spatial_size);

template <typename T>
__global__ void compute_convergence_metric(
    const T *spatial_error,
    T spatial_epsilon,
    T *convergence_result,
    size_t total_size);

template <typename T>
__global__ void quantize_and_dequantize(
    const T *org_arr,
    T *dequant_arr,
    T min_val,
    T max_val,
    size_t total_size,
    T threshold = T(1e-7));

template <typename T>
__global__ void frequency_float_to_complex(
    const T *freq_split,
    typename CufftTraits<T>::ComplexType *freq_complex,
    size_t total_size);

template <typename T>
__global__ void add_two_arrays(
    const T *arr1,
    T *arr2,
    size_t total_size);

// Host function declarations
template <typename T>
class ProjectionSolver
{
public:
    ProjectionSolver(size_t Nx, size_t Ny, size_t Nz);
    ~ProjectionSolver();

    void initialize(
        const T *h_orig_data,
        const size_t *h_freq_indices,
        const T *h_delta_k,
        T spatial_epsilon,
        size_t num_freq_constraints);

    void initialize_all_abs(
        const T *h_orig_data,
        T spatial_epsilon,
        T delta);

    void initialize_all_ptw(
        const T *h_orig_data,
        T spatial_epsilon,
        T delta_ptw);

    void initialize_partial_ptw(
        const T *h_orig_data,
        const size_t *h_freq_indices,
        const T delta_ptw,
        T spatial_epsilon,
        size_t num_freq_constraints);

    void solve(
        T *h_spatial_error,
        int max_iterations,
        T convergence_tolerance);

    void solve_all_abs(
        T *h_spatial_error,
        int max_iterations,
        T convergence_tolerance);

    void solve_all_ptw(
        T *h_spatial_error,
        int max_iterations,
        T convergence_tolerance);

    void get_results(T *h_spatial_error);

    // Methods to extract compact results from accumulators
    void extract_compact_representation();
    std::vector<bool> get_freq_flags();
    std::vector<bool> get_spatial_flags();
    std::vector<uint8_t> get_freq_flags_pack();
    std::vector<uint8_t> get_spatial_flags_pack();
    std::vector<uint16_t> get_active_freq_quant_edit_distances();
    std::vector<uint16_t> get_active_spatial_quant_edit_distances();
    std::vector<T> get_active_freq_edit_distances();
    std::vector<T> get_active_spatial_edit_distances();

    // Methods to get full arrays
    std::vector<T> get_freq_edits_full();    // Full accumulator array
    std::vector<T> get_spatial_edits_full(); // Full accumulator array

    // Methods to get statistics of compact results
    T get_min_freq_edit();
    T get_max_freq_edit();
    T get_min_spatial_edit();
    T get_max_spatial_edit();

    // Reset accumulators
    void reset_edit_accumulators();

private:
    ProjectionParams<T> params_;
    IterationResults<T> results_;
    bool initialized_;

    // Temporary buffers for compaction
    size_t *d_temp_write_pos_;

    void setup_fft_plans();
    void cleanup_fft_plans();
    void allocate_device_memory();
    void deallocate_device_memory();

    void perform_single_iteration();
    void project_onto_frequency_polytope();
    void project_onto_frequency_polytope_all_abs();
    void project_onto_frequency_polytope_all_ptw();
    void project_onto_frequency_polytope_partial_ptw();
    void project_onto_spatial_box();
    T compute_convergence();

    // Common initialization helper
    void initialize_common(const T *h_orig_data, T spatial_epsilon);
};

// Include implementation
#include "projection_algorithm_impl.cuh"