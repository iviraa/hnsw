#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "dataset.h"

// Exact search, used both as ground truth and as the speed baseline.
std::vector<std::pair<float, uint32_t>> brute_force(const float* query, const Dataset& ds, size_t k);

std::vector<std::vector<uint32_t>> exact_neighbours(const Dataset& queries, const Dataset& base,
                                                    size_t k);

// Fraction of the true k neighbours returned, ignoring order.
float recall(const std::vector<uint32_t>& got, const std::vector<uint32_t>& truth, size_t k);
