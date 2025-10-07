#include "fileIO.h"

void writeMap(const std::map<int, std::pair<uint32_t, int>> &m, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    size_t size = m.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (const auto &[key, val] : m)
    {
        ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
        ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
        ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
    }
}

std::map<int, std::pair<uint32_t, int>> readMap(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    std::map<int, std::pair<uint32_t, int>> m;
    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i)
    {
        int key;
        uint32_t first;
        int second;
        ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
        ifs.read(reinterpret_cast<char *>(&first), sizeof(first));
        ifs.read(reinterpret_cast<char *>(&second), sizeof(second));
        m[key] = {first, second};
    }
    return m;
}

void readRawArrayBinary(const std::string &fileName, void *data, std::size_t N, DataType type)
{
    std::ifstream inputFile(fileName, std::ios::binary | std::ios::ate);
    if (!inputFile)
    {
        throw std::runtime_error("Error opening file");
    }

    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    std::size_t elementSize;
    switch (type)
    {
    case DataType::FLOAT:
        elementSize = sizeof(float);
        break;
    case DataType::DOUBLE:
        elementSize = sizeof(double);
        break;
    case DataType::SIZE_T:
        elementSize = sizeof(size_t);
        break;
    case DataType::INT:
        elementSize = sizeof(int);
        break;
    default:
        throw std::invalid_argument("Invalid data type");
    }

    if (fileSize != static_cast<std::streamsize>(elementSize * N))
    {
        throw std::runtime_error("File size does not match array size");
    }

    if (!inputFile.read(reinterpret_cast<char *>(data), fileSize))
    {
        throw std::runtime_error("Error reading file");
    }

    inputFile.close();
}

template void writeVectorBinary(const std::vector<uint8_t> &vec, const std::string &filename);
template void writeVectorBinary(const std::vector<uint16_t> &vec, const std::string &filename);
template void writeVectorBinary(const std::vector<size_t> &vec, const std::string &filename);
template void writeVectorBinary(const std::vector<float> &vec, const std::string &filename);
template void writeVectorBinary(const std::vector<double> &vec, const std::string &filename);

template void writeRawArrayBinary(const float *data, size_t N, const std::string &filename);
template void writeRawArrayBinary(const double *data, size_t N, const std::string &filename);