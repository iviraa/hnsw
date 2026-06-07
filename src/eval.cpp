#include "eval.h"

#include <algorithm>
#include <unordered_set>

#include "distance.h"

std::vector<std::pair<float, uint32_t>> brute_force(const float* query, const Dataset& ds,
                                                    size_t k) {
    std::vector<std::pair<float, uint32_t>> all;
    all.reserve(ds.size());
    for (size_t i = 0; i < ds.size(); i++)
        all.push_back({l2_simd(query, ds.at(i), ds.dim), (uint32_t)i});

    size_t want = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + want, all.end());
    all.resize(want);
    return all;
}

std::vector<std::vector<uint32_t>> exact_neighbours(const Dataset& queries, const Dataset& base,
                                                    size_t k) {
    std::vector<std::vector<uint32_t>> out;
    out.reserve(queries.size());
    for (size_t i = 0; i < queries.size(); i++) {
        auto hits = brute_force(queries.at(i), base, k);
        std::vector<uint32_t> ids;
        for (auto& h : hits) ids.push_back(h.second);
        out.push_back(ids);
    }
    return out;
}

float recall(const std::vector<uint32_t>& got, const std::vector<uint32_t>& truth, size_t k) {
    size_t want = std::min(k, truth.size());
    if (want == 0) return 1.0f;

    std::unordered_set<uint32_t> expected(truth.begin(), truth.begin() + want);
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, got.size()); i++)
        if (expected.count(got[i])) hits++;
    return (float)hits / want;
}
