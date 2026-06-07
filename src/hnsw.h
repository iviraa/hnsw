#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dataset.h"

// Layered proximity graph. Search walks it greedily, starting from the sparse top.

struct Params {
    size_t M = 16;                // links per node above layer 0
    size_t ef_construction = 200; // search width while building
    size_t ef_search = 50;        // search width when querying
    bool simd = true;
    bool use_heuristic = true;
    bool hierarchical = true;     // false gives the flat NSW graph this came from
    uint64_t seed = 42;
};

class Hnsw {
public:
    // Distance first so the default pair comparison sorts by it.
    using Hit = std::pair<float, uint32_t>;

    Hnsw(size_t dim, Params p);
    ~Hnsw();
    Hnsw(Hnsw&&) noexcept;
    Hnsw& operator=(Hnsw&&) noexcept;
    Hnsw(const Hnsw&) = delete;

    void build(const Dataset& ds);
    void build_threaded(const Dataset& ds, unsigned threads = 0);

    std::vector<Hit> search(const float* query, size_t k) const;
    std::vector<Hit> search(const float* query, size_t k, size_t ef) const;

    void save(const std::string& path) const;
    static Hnsw load(const std::string& path);
    static Hnsw load_mapped(const std::string& path);

    size_t size() const { return n; }
    int top_level() const { return max_level; }
    size_t memory_bytes() const;
    float mean_degree() const;

    // Used by the tools that draw the graph.
    int level_of(uint32_t node) const { return levels[node]; }
    uint32_t entry() const { return entry_point; }
    std::vector<uint32_t> neighbours(uint32_t node, int level) const;

private:
    float distance(const float* query, uint32_t node) const;
    float distance(uint32_t a, uint32_t b) const;
    const float* vec(uint32_t id) const { return vectors + (size_t)id * dim; }

    // Fixed slots per node, count in slot 0. Upper layers are packed, see link_offset.
    uint32_t* links(uint32_t node, int level);
    const uint32_t* links(uint32_t node, int level) const;
    size_t max_links(int level) const { return level == 0 ? m0 : params.M; }
    const uint32_t* read_links(uint32_t node, int level, uint32_t& count) const;

    std::vector<int> prepare(const Dataset& ds);
    void insert(uint32_t id, const float* v, int level);
    uint32_t descend(const float* query, uint32_t entry, float& dist, int level) const;
    void search_layer(const float* query, uint32_t entry, float dist, size_t ef, int level,
                      std::vector<Hit>& found) const;
    void select_neighbours(std::vector<Hit>& candidates, size_t want,
                           std::vector<uint32_t>& out) const;
    void connect(uint32_t node, const std::vector<uint32_t>& neighbours, int level);
    void refresh_pointers();

    size_t dim;
    Params params;
    size_t m0;      // 2 * M, layer 0 gets twice the links
    double level_scale;

    std::vector<float> vector_store;
    std::vector<int32_t> node_level;
    std::vector<uint32_t> layer0;
    std::vector<uint32_t> upper;
    std::vector<uint64_t> link_offset;

    // Reads go through these, so a mapped index and a built one share one path.
    const float* vectors = nullptr;
    const int32_t* levels = nullptr;
    uint32_t* layer0_p = nullptr;
    uint32_t* upper_p = nullptr;
    const uint64_t* offset_p = nullptr;

    void* mapping = nullptr;
    size_t mapping_size = 0;

    uint32_t n = 0;
    uint32_t entry_point = 0;
    int max_level = -1;

    struct Locks;
    std::unique_ptr<Locks> locks;
    bool building_threaded = false;
};
