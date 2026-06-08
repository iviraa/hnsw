#pragma once
// Shared setup so the SVG and the animation show the same graph.
#include <cstdint>

#include "dataset.h"
#include "hnsw.h"

// Spread evenly, since the clustered generator bunches everything into blobs.
inline Dataset spread_points(size_t n, uint64_t seed) {
    uint64_t state = seed * 2 + 1;
    auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)(state >> 40) / 16777216.0f;
    };

    Dataset ds;
    ds.dim = 2;
    ds.data.resize(n * 2);
    for (size_t i = 0; i < n * 2; i++) ds.data[i] = next();
    return ds;
}

inline Params demo_params() {
    Params p;
    p.M = 6;  // small so the edges stay readable
    p.ef_construction = 60;
    return p;
}
