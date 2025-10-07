#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>
#include <algorithm>
#include "projection_algorithm.cuh"
#include "fileIO.h"
// #include "HuffmanCoder.h"
#include "HuffmanZSTDCoder.cuh"

std::string originalFile;
std::string baseDecompFile;
std::string compressedFile;
std::string decompressedFile;
size_t Nx = 0;
size_t Ny = 0;
size_t Nz = 1;
size_t spatial_size = 0;
// size_t k = 0;
bool isDouble = false;
bool isSpatialABS = false;
bool isFreqABS = false;
bool isMaxBound = false;
float spatial_epsilon;
float freq_delta; // component-wise relative error

void parseError(const char error[])
{
    std::cout << error << std::endl;
    std::cout << "Usage:\n";
    std::cout << "  -i <file_path>  : Specify the file of original data\n";
    std::cout << "  -e <file_path>  : Specify the reconstructed file of base compressor\n";
    std::cout << "  -z <file_path>  : Specify the compressed file\n";
    std::cout << "  -o <file_path>  : Specify the decompressed file (optional)\n";
    std::cout << "  -1 <nx>                : 1D data with <nx> values\n";
    std::cout << "  -2 <nx> <ny>           : 2D data with <nx> * <ny> values\n";
    std::cout << "  -3 <nx> <ny> <nz>      : 3D data with <nx> * <ny> * <nz> values\n";
    std::cout << "  -f              : Use float data type\n";
    std::cout << "  -d              : Use double data type\n";
    std::cout << "  -M ABS <xi>     : Specify the absolute error bound in spatial domain\n";
    std::cout << "  -M REL <xi>     : Specify the relative error bound in spatial domain\n";
    std::cout << "  -F ABS <xi>     : Specify the absolute error bound in frequency domain\n";
    std::cout << "  -F REL <xi>     : Specify the relative error bound in frequency domain\n";
    // std::cout << "  -k MAX <k>      : Specify the maximum frequency to be bounded in power spectrum (optional)\n";
    // std::cout << "  -k MIN <k>      : Specify the minimum frequency to be bounded in power spectrum (optional)\n";
    exit(EXIT_FAILURE);
}

void Parsing(int argc, char *argv[])
{
    bool originalFileSpecified = false;
    bool baseDecompFileSpecified = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-i")
        {
            if (i + 1 >= argc)
                parseError("Missing input file path");
            originalFile = argv[++i];
            originalFileSpecified = true;
        }
        else if (arg == "-e")
        {
            if (i + 1 >= argc)
                parseError("Missing base reconstructed file path");
            baseDecompFile = argv[++i];
            baseDecompFileSpecified = true;
        }
        else if (arg == "-z")
        {
            if (i + 1 >= argc)
                parseError("Missing compressed file path");
            compressedFile = argv[++i];
        }
        else if (arg == "-o")
        {
            if (i + 1 >= argc)
                parseError("Missing decompressed file path");
            decompressedFile = argv[++i];
        }
        else if (arg == "-1" || arg == "-2" || arg == "-3")
        {
            if (arg == "-1")
            {
                Nx = 1;
                Ny = 1;
                Nz = std::stoi(argv[++i]);
                spatial_size = Nz;
            }
            else if (arg == "-2")
            {
                Nx = 1;
                Ny = std::stoi(argv[++i]);
                Nz = std::stoi(argv[++i]);
                spatial_size = Ny * Nz;
            }
            else if (arg == "-3")
            {
                Nx = std::stoi(argv[++i]);
                Ny = std::stoi(argv[++i]);
                Nz = std::stoi(argv[++i]);
                spatial_size = Nx * Ny * Nz;
            }

            if (i + 1 >= argc)
            {
                parseError("Missing size for 1D data");
            }
        }
        else if (arg == "-f")
        {
            isDouble = false; // Use float type
        }
        else if (arg == "-d")
        {
            isDouble = true; // Use double type
        }
        else if (arg == "-M")
        {
            isSpatialABS = std::strcmp(argv[++i], "ABS") == 0;
            if (i + 1 >= argc)
                parseError("Missing spatial error bound");
            spatial_epsilon = std::stof(argv[++i]);
        }
        else if (arg == "-F")
        {
            isFreqABS = std::strcmp(argv[++i], "ABS") == 0;
            if (i + 1 >= argc)
                parseError("Missing frequency error bound");
            freq_delta = std::stof(argv[++i]);
        }
        // else if (arg == "-k")
        // {
        //     isMaxBound = std::strcmp(argv[++i], "MAX") == 0;
        //     if (i + 1 >= argc)
        //         parseError("Missing relative error bound");
        //     k = std::stoi(argv[++i]);
        // }
        else
        {
            parseError("Unknown argument");
        }
    }

    if (!originalFileSpecified && !baseDecompFileSpecified)
    {
        parseError("Input files of original data (-i) and base decompressed data (-e) are mandatory");
    }
}

template <typename T>
void Run()
{
    // Original data
    T *h_original_data = new T[spatial_size];
    T *h_spatial_error = new T[spatial_size];
    if constexpr (std::is_same_v<T, float>)
    {
        readRawArrayBinary(originalFile, h_original_data, spatial_size, DataType::FLOAT);
        readRawArrayBinary(baseDecompFile, h_spatial_error, spatial_size, DataType::FLOAT);
    }
    else
    {
        readRawArrayBinary(originalFile, h_original_data, spatial_size, DataType::DOUBLE);
        readRawArrayBinary(baseDecompFile, h_spatial_error, spatial_size, DataType::DOUBLE);
    }
    for (size_t i = 0; i < spatial_size; ++i)
    {
        h_spatial_error[i] -= h_original_data[i];
    }

    // Initialize solver
    ProjectionSolver<T> solver(Nx, Ny, Nz);

    // solver.initialize(
    //     h_original_data,
    //     h_freq_indices,
    //     h_delta_k,
    //     spatial_epsilon,
    //     num_freq_constraints);

    solver.initialize_all_abs(
        h_original_data,
        spatial_epsilon,
        freq_delta);

    // solver.initialize_all_ptw(
    //     h_original_data,
    //     spatial_epsilon,
    //     freq_delta);

    // Solve the projection problem
    const int max_iterations = 100;
    const T tolerance = 1e-7f;

    auto start_time = std::chrono::high_resolution_clock::now();

    // solver.solve(
    //     h_spatial_error,
    //     max_iterations,
    //     tolerance);

    solver.solve_all_abs(
        h_spatial_error,
        max_iterations,
        tolerance);

    // solver.solve_all_ptw(
    //     h_spatial_error,
    //     max_iterations,
    //     tolerance);

    auto mid_time = std::chrono::high_resolution_clock::now();

    // Get the compact edits (distance) and pack the flags
    std::vector<uint16_t> active_freq_distances = solver.get_active_freq_quant_edit_distances();
    std::vector<uint16_t> active_spatial_distances = solver.get_active_spatial_quant_edit_distances();
    std::vector<uint8_t> packed_freq_flags = solver.get_freq_flags_pack();
    std::vector<uint8_t> packed_spatial_flags = solver.get_spatial_flags_pack();

    auto end_time = std::chrono::high_resolution_clock::now();

    // Lossless compression
    size_t extra_storage = 0;
    size_t num_active_freq = active_freq_distances.size();
    size_t num_active_spatial = active_spatial_distances.size();
    if (num_active_freq > 0)
    {
        auto compressed_edits = cudaHuffmanZstdCompress(active_freq_distances, 20);
        extra_storage += compressed_edits.compressed_data.size();
        auto compressed_flags = cudaHuffmanZstdCompress(packed_freq_flags, 20);
        extra_storage += compressed_flags.compressed_data.size(); // storage of max and min of original array
    }
    if (num_active_spatial > 0)
    {
        auto compressed_edits = cudaHuffmanZstdCompress(active_spatial_distances, 20);
        extra_storage += compressed_edits.compressed_data.size();
        auto compressed_flags = cudaHuffmanZstdCompress(packed_spatial_flags, 20);
        extra_storage += compressed_flags.compressed_data.size();
    }
    printf("# Active frequency edits: %zu\n", num_active_freq);
    printf("# Active spatial edits: %zu\n", num_active_spatial);
    // printf("Extra storage: %zu bytes\n", extra_storage);

    auto compression_time = std::chrono::duration_cast<std::chrono::milliseconds>((end_time - mid_time) * 3 + mid_time - start_time);
    auto decompression_time = std::chrono::duration_cast<std::chrono::milliseconds>((end_time - mid_time) * 2);
    std::cout << "Compression time: " << compression_time.count() << " ms" << std::endl;
    std::cout << "Decompression time: " << decompression_time.count() << " ms" << std::endl;

    // solver.get_results(h_spatial_error);
    // writeRawArrayBinary(h_spatial_error, spatial_size, decompressedFile);

    // // Get other results
    // std::vector<float> freq_edits = solver.get_freq_edits_full();
    // std::vector<float> spat_edits = solver.get_spatial_edits_full();
    // std::vector<bool> freq_flags = solver.get_freq_flags();
    // std::vector<bool> spatial_flags = solver.get_spatial_flags();
    // std::vector<float> active_freq_dists = solver.get_active_freq_edit_distances();
    // std::vector<float> active_spatial_dists = solver.get_active_spatial_edit_distances();
}

int main(int argc, char *argv[])
{
    Parsing(argc, argv);

    if (isDouble)
    {
        Run<double>();
    }
    else
    {
        Run<float>();
    }
}