#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <zstd.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        }                                                                                    \
    } while (0)

template <typename T>
struct HuffmanNodeGPU
{
    T value;
    uint32_t freq;
    int left, right;

    __host__ __device__
    HuffmanNodeGPU() : value(), freq(0), left(-1), right(-1) {}

    __host__ __device__
    HuffmanNodeGPU(T val, uint32_t f) : value(val), freq(f), left(-1), right(-1) {}

    __host__ __device__ bool is_leaf() const { return left == -1 && right == -1; }
};

// Simple GPU frequency counting using sort + reduce_by_key
template <typename T>
void cudaFrequencyCount(const thrust::device_vector<T> &data,
                        thrust::device_vector<T> &unique_values,
                        thrust::device_vector<uint32_t> &frequencies)
{
    if (data.empty())
        return;

    // Sort data to group identical values
    thrust::device_vector<T> sorted_data = data;
    thrust::sort(thrust::device, sorted_data.begin(), sorted_data.end());

    // Create counting iterator for reduce_by_key
    thrust::device_vector<uint32_t> ones(data.size(), 1);

    // Allocate space for results
    thrust::device_vector<T> temp_keys(data.size());
    thrust::device_vector<uint32_t> temp_counts(data.size());

    // Use reduce_by_key to count frequencies
    auto end_pair = thrust::reduce_by_key(
        thrust::device,
        sorted_data.begin(), sorted_data.end(),
        ones.begin(),
        temp_keys.begin(),
        temp_counts.begin());

    size_t unique_count = end_pair.first - temp_keys.begin();

    unique_values.resize(unique_count);
    frequencies.resize(unique_count);

    thrust::copy(thrust::device, temp_keys.begin(), temp_keys.begin() + unique_count, unique_values.begin());
    thrust::copy(thrust::device, temp_counts.begin(), temp_counts.begin() + unique_count, frequencies.begin());
}

// Simple CPU-based Huffman tree construction
template <typename T>
void buildHuffmanTree(const thrust::device_vector<T> &symbols,
                      const thrust::device_vector<uint32_t> &frequencies,
                      thrust::device_vector<HuffmanNodeGPU<T>> &nodes,
                      int &root_idx)
{
    thrust::host_vector<T> h_symbols = symbols;
    thrust::host_vector<uint32_t> h_frequencies = frequencies;

    int n = h_symbols.size();
    if (n == 0)
    {
        root_idx = -1;
        return;
    }

    if (n == 1)
    {
        // Special case: single symbol
        nodes.resize(1);
        thrust::host_vector<HuffmanNodeGPU<T>> h_nodes(1);
        h_nodes[0] = HuffmanNodeGPU<T>(h_symbols[0], h_frequencies[0]);
        nodes = h_nodes;
        root_idx = 0;
        return;
    }

    // Create priority queue using vector and make_heap
    std::vector<std::pair<uint32_t, int>> pq; // (frequency, node_index)
    thrust::host_vector<HuffmanNodeGPU<T>> h_nodes(2 * n - 1);

    // Initialize leaf nodes
    for (int i = 0; i < n; ++i)
    {
        h_nodes[i] = HuffmanNodeGPU<T>(h_symbols[i], h_frequencies[i]);
        pq.push_back({h_frequencies[i], i});
    }

    std::make_heap(pq.begin(), pq.end(), std::greater<std::pair<uint32_t, int>>());

    int next_node_idx = n;

    while (pq.size() > 1)
    {
        // Get two minimum nodes
        std::pop_heap(pq.begin(), pq.end(), std::greater<std::pair<uint32_t, int>>());
        auto min1 = pq.back();
        pq.pop_back();

        std::pop_heap(pq.begin(), pq.end(), std::greater<std::pair<uint32_t, int>>());
        auto min2 = pq.back();
        pq.pop_back();

        // Create new internal node
        h_nodes[next_node_idx].freq = min1.first + min2.first;
        h_nodes[next_node_idx].left = min1.second;
        h_nodes[next_node_idx].right = min2.second;

        pq.push_back({h_nodes[next_node_idx].freq, next_node_idx});
        std::push_heap(pq.begin(), pq.end(), std::greater<std::pair<uint32_t, int>>());

        ++next_node_idx;
    }

    root_idx = pq[0].second;
    nodes = h_nodes;
}

// Generate Huffman codes recursively
template <typename T>
void generateCodesRecursive(const thrust::host_vector<HuffmanNodeGPU<T>> &nodes,
                            int node_idx,
                            std::unordered_map<T, std::pair<uint32_t, int>> &code_table,
                            uint32_t code, int length)
{
    if (node_idx == -1)
        return;

    const HuffmanNodeGPU<T> &node = nodes[node_idx];

    if (node.is_leaf())
    {
        // Handle single symbol case
        if (length == 0)
            length = 1;
        code_table[node.value] = {code, length};
    }
    else
    {
        generateCodesRecursive(nodes, node.left, code_table, code << 1, length + 1);
        generateCodesRecursive(nodes, node.right, code_table, (code << 1) | 1, length + 1);
    }
}

// GPU bit packing kernel
template <typename T>
__global__ void packBitsKernel(const T *data, size_t data_size,
                               const T *symbols, const uint32_t *codes,
                               const int *lengths, size_t num_symbols,
                               uint64_t *output, size_t *bit_positions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= data_size)
        return;

    T value = data[tid];

    // Linear search for symbol (could be optimized with binary search)
    uint32_t code = 0;
    int length = 1; // Default length

    for (size_t i = 0; i < num_symbols; ++i)
    {
        if (symbols[i] == value)
        {
            code = codes[i];
            length = lengths[i];
            break;
        }
    }

    size_t bit_pos = bit_positions[tid];
    size_t word_idx = bit_pos / 64;
    int bit_offset = bit_pos % 64;

    // Pack bits into 64-bit words
    if (bit_offset + length <= 64)
    {
        uint64_t shifted_code = (uint64_t)code << (64 - bit_offset - length);
        atomicOr((unsigned long long *)&output[word_idx], (unsigned long long)shifted_code);
    }
    else
    {
        // Handle word boundary crossing
        int first_part = 64 - bit_offset;
        int second_part = length - first_part;

        uint64_t first_bits = (uint64_t)(code >> second_part);
        uint64_t second_bits = (uint64_t)(code & ((1ULL << second_part) - 1));

        atomicOr((unsigned long long *)&output[word_idx], (unsigned long long)first_bits);
        if (second_part > 0)
        {
            atomicOr((unsigned long long *)&output[word_idx + 1],
                     (unsigned long long)(second_bits << (64 - second_part)));
        }
    }
}

// Main CUDA compression class
template <typename T>
class SimpleCudaHuffmanZstdCompressor
{
public:
    struct CompressionResult
    {
        std::vector<uint8_t> compressed_data;
        std::unordered_map<T, std::pair<uint32_t, int>> code_table;
        size_t original_bitstream_size;
        size_t num_elements;
    };

    CompressionResult compress(const std::vector<T> &data, int zstd_level = 3)
    {
        if (data.empty())
        {
            throw std::runtime_error("Empty data");
        }

        // Copy data to GPU
        thrust::device_vector<T> d_data = data;

        // 1. GPU frequency counting
        thrust::device_vector<T> d_unique_symbols;
        thrust::device_vector<uint32_t> d_frequencies;
        cudaFrequencyCount(d_data, d_unique_symbols, d_frequencies);

        size_t num_symbols = d_unique_symbols.size();
        if (num_symbols == 0)
        {
            throw std::runtime_error("No symbols found");
        }

        // 2. Build Huffman tree on CPU
        thrust::device_vector<HuffmanNodeGPU<T>> d_nodes;
        int root_idx;
        buildHuffmanTree(d_unique_symbols, d_frequencies, d_nodes, root_idx);

        // 3. Generate codes
        thrust::host_vector<HuffmanNodeGPU<T>> h_nodes = d_nodes;
        std::unordered_map<T, std::pair<uint32_t, int>> code_table;
        generateCodesRecursive(h_nodes, root_idx, code_table, 0, 0);

        // 4. Prepare GPU arrays for bit packing
        thrust::host_vector<T> h_symbols = d_unique_symbols;
        thrust::device_vector<T> d_symbols = d_unique_symbols;
        thrust::device_vector<uint32_t> d_codes(num_symbols);
        thrust::device_vector<int> d_lengths(num_symbols);

        thrust::host_vector<uint32_t> h_codes(num_symbols);
        thrust::host_vector<int> h_lengths(num_symbols);

        for (size_t i = 0; i < num_symbols; ++i)
        {
            T symbol = h_symbols[i];
            auto it = code_table.find(symbol);
            if (it != code_table.end())
            {
                h_codes[i] = it->second.first;
                h_lengths[i] = it->second.second;
            }
            else
            {
                h_codes[i] = 0;
                h_lengths[i] = 1;
            }
        }

        d_codes = h_codes;
        d_lengths = h_lengths;

        // 5. Calculate bit positions using CPU (simpler and more reliable)
        std::vector<size_t> bit_positions(data.size());
        size_t total_bits = 0;

        for (size_t i = 0; i < data.size(); ++i)
        {
            bit_positions[i] = total_bits;

            // Find symbol length
            int length = 1; // default
            auto it = code_table.find(data[i]);
            if (it != code_table.end())
            {
                length = it->second.second;
            }
            total_bits += length;
        }

        thrust::device_vector<size_t> d_bit_positions = bit_positions;

        // 6. Pack bits on GPU
        size_t num_words = (total_bits + 63) / 64;
        thrust::device_vector<uint64_t> d_packed(num_words, 0);

        dim3 block(256);
        dim3 grid((data.size() + block.x - 1) / block.x);

        packBitsKernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_data.data()), data.size(),
            thrust::raw_pointer_cast(d_symbols.data()),
            thrust::raw_pointer_cast(d_codes.data()),
            thrust::raw_pointer_cast(d_lengths.data()), num_symbols,
            thrust::raw_pointer_cast(d_packed.data()),
            thrust::raw_pointer_cast(d_bit_positions.data()));

        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. Convert to bytes
        thrust::host_vector<uint64_t> h_packed = d_packed;
        std::vector<uint8_t> bit_stream;
        size_t total_bytes = (total_bits + 7) / 8;
        bit_stream.reserve(total_bytes);

        for (size_t i = 0; i < num_words && bit_stream.size() < total_bytes; ++i)
        {
            uint64_t word = h_packed[i];
            for (int j = 7; j >= 0 && bit_stream.size() < total_bytes; --j)
            {
                bit_stream.push_back((word >> (j * 8)) & 0xFF);
            }
        }

        // 8. ZSTD compression
        size_t bound = ZSTD_compressBound(bit_stream.size());
        std::vector<uint8_t> compressed(bound);

        size_t compressed_size = ZSTD_compress(compressed.data(), bound,
                                               bit_stream.data(), bit_stream.size(),
                                               zstd_level);

        if (ZSTD_isError(compressed_size))
        {
            throw std::runtime_error("ZSTD compression failed: " +
                                     std::string(ZSTD_getErrorName(compressed_size)));
        }

        compressed.resize(compressed_size);

        return {compressed, code_table, bit_stream.size(), data.size()};
    }

    std::vector<T> decompress(const std::vector<uint8_t> &compressed_data,
                              const std::unordered_map<T, std::pair<uint32_t, int>> &code_table,
                              size_t num_elements,
                              size_t original_bitstream_size)
    {
        // ZSTD decompression
        std::vector<uint8_t> bit_stream(original_bitstream_size);
        size_t decompressed_size = ZSTD_decompress(bit_stream.data(), original_bitstream_size,
                                                   compressed_data.data(), compressed_data.size());

        if (ZSTD_isError(decompressed_size))
        {
            throw std::runtime_error("ZSTD decompression failed: " +
                                     std::string(ZSTD_getErrorName(decompressed_size)));
        }

        // Build reverse lookup table
        std::unordered_map<uint64_t, T> reverse_table;
        for (const auto &[symbol, code_len] : code_table)
        {
            uint32_t code = code_len.first;
            int length = code_len.second;
            uint64_t key = ((uint64_t)length << 32) | code;
            reverse_table[key] = symbol;
        }

        std::vector<T> result;
        result.reserve(num_elements);

        size_t bit_pos = 0;
        size_t total_bits = bit_stream.size() * 8;

        while (result.size() < num_elements && bit_pos < total_bits)
        {
            bool found = false;

            // Try different code lengths
            for (int len = 1; len <= 32 && bit_pos + len <= total_bits && !found; ++len)
            {
                uint32_t code = 0;

                // Extract 'len' bits
                for (int i = 0; i < len; ++i)
                {
                    size_t byte_idx = (bit_pos + i) / 8;
                    int bit_idx = 7 - ((bit_pos + i) % 8);

                    if (byte_idx < bit_stream.size())
                    {
                        int bit = (bit_stream[byte_idx] >> bit_idx) & 1;
                        code = (code << 1) | bit;
                    }
                }

                uint64_t key = ((uint64_t)len << 32) | code;
                auto it = reverse_table.find(key);
                if (it != reverse_table.end())
                {
                    result.push_back(it->second);
                    bit_pos += len;
                    found = true;
                }
            }

            if (!found)
            {
                // Fallback: skip bit
                bit_pos++;
            }
        }

        return result;
    }
};

// Convenience functions
template <typename T>
typename SimpleCudaHuffmanZstdCompressor<T>::CompressionResult
cudaHuffmanZstdCompress(const std::vector<T> &data, int zstd_level = 3)
{
    SimpleCudaHuffmanZstdCompressor<T> compressor;
    return compressor.compress(data, zstd_level);
}

template <typename T>
std::vector<T> cudaHuffmanZstdDecompress(const std::vector<uint8_t> &compressed_data,
                                         const std::unordered_map<T, std::pair<uint32_t, int>> &code_table,
                                         size_t num_elements,
                                         size_t original_bitstream_size)
{
    SimpleCudaHuffmanZstdCompressor<T> compressor;
    return compressor.decompress(compressed_data, code_table, num_elements, original_bitstream_size);
}

// // Test function
// void testCompression()
// {
//     std::vector<int> test_data = {1, 2, 3, 1, 2, 1, 4, 5, 1, 2, 3, 3, 3};

//     std::cout << "Original data size: " << test_data.size() << " elements\n";

//     try
//     {
//         auto result = cudaHuffmanZstdCompress(test_data, 3);

//         std::cout << "Compressed size: " << result.compressed_data.size() << " bytes\n";
//         std::cout << "Bitstream size: " << result.original_bitstream_size << " bytes\n";
//         std::cout << "Code table size: " << result.code_table.size() << " symbols\n";

//         // Print code table
//         for (const auto &[symbol, code_len] : result.code_table)
//         {
//             std::cout << "Symbol " << symbol << ": code=" << code_len.first
//                       << ", length=" << code_len.second << "\n";
//         }

//         auto decompressed = cudaHuffmanZstdDecompress(
//             result.compressed_data,
//             result.code_table,
//             result.num_elements,
//             result.original_bitstream_size);

//         bool success = (test_data == decompressed);
//         std::cout << "Decompression successful: " << (success ? "YES" : "NO") << "\n";

//         if (!success)
//         {
//             std::cout << "Original:     ";
//             for (int x : test_data)
//                 std::cout << x << " ";
//             std::cout << "\nDecompressed: ";
//             for (int x : decompressed)
//                 std::cout << x << " ";
//             std::cout << "\n";
//         }
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "Error: " << e.what() << "\n";
//     }
// }