#pragma once
#include <cstdint>
#include <string>
#include <vector>

// All vectors in one array, vector i starting at data[i * dim].
struct Dataset {
    std::vector<float> data;
    size_t dim = 0;

    size_t size() const { return dim ? data.size() / dim : 0; }
    const float* at(size_t i) const { return data.data() + i * dim; }
};

// SIFT format: each vector is a 4 byte dimension then that many values.
Dataset read_fvecs(const std::string& path);
std::vector<std::vector<uint32_t>> read_ivecs(const std::string& path);

// Clustered, because uniform points in high dimensions sit at near equal distances.
Dataset make_test_data(size_t n, size_t dim, size_t clusters, uint64_t seed);
