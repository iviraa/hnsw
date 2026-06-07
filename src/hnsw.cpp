#include "hnsw.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "distance.h"

// Shared pool, since one mutex per node would cost gigabytes at a million nodes.
static const size_t kLockCount = 4096;

struct Hnsw::Locks {
    std::vector<std::shared_mutex> node{kLockCount};
    std::mutex global;
};

namespace {

// Candidates come out nearest first, results worst first so eviction is cheap.
struct Nearest {
    bool operator()(const Hnsw::Hit& a, const Hnsw::Hit& b) const { return a.first > b.first; }
};
struct Furthest {
    bool operator()(const Hnsw::Hit& a, const Hnsw::Hit& b) const { return a.first < b.first; }
};

// Stamped rather than cleared, so nothing has to be reset between queries.
struct Visited {
    std::vector<uint32_t> stamp;
    uint32_t current = 0;

    uint32_t begin(size_t n) {
        if (stamp.size() < n) stamp.resize(n, 0);
        if (++current == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            current = 1;
        }
        return current;
    }
};

thread_local Visited visited;
thread_local std::vector<Hnsw::Hit> candidates;
thread_local std::vector<Hnsw::Hit> found_scratch;
thread_local std::vector<uint32_t> neighbour_copy;
thread_local std::vector<uint32_t> chosen;

struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed * 2 + 1) {}

    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }
    double uniform() { return (next() >> 11) / 9007199254740992.0; }
};

const uint32_t kMagic = 0x56534831;  // "VSH1"

struct Header {
    uint32_t magic;
    uint64_t dim;
    Params params;
    uint32_t n;
    uint32_t entry;
    int32_t max_level;
    uint64_t upper_len;
    uint64_t off_vectors;
    uint64_t off_levels;
    uint64_t off_layer0;
    uint64_t off_offsets;
    uint64_t off_upper;
    uint64_t total;
};

uint64_t round_up(uint64_t x) { return (x + 63) & ~63ULL; }

}  // namespace

Hnsw::Hnsw(size_t d, Params p)
    : dim(d), params(p), m0(p.M * 2), level_scale(1.0 / std::log((double)p.M)) {
    if (p.M < 2) throw std::invalid_argument("M must be at least 2");
    link_offset.push_back(0);
}

Hnsw::~Hnsw() {
    if (mapping) munmap(mapping, mapping_size);
}

Hnsw::Hnsw(Hnsw&& o) noexcept : dim(0), params(), m0(0), level_scale(0) { *this = std::move(o); }

Hnsw& Hnsw::operator=(Hnsw&& o) noexcept {
    if (this == &o) return *this;
    if (mapping) munmap(mapping, mapping_size);

    dim = o.dim;
    params = o.params;
    m0 = o.m0;
    level_scale = o.level_scale;
    vector_store = std::move(o.vector_store);
    node_level = std::move(o.node_level);
    layer0 = std::move(o.layer0);
    upper = std::move(o.upper);
    link_offset = std::move(o.link_offset);
    n = o.n;
    entry_point = o.entry_point;
    max_level = o.max_level;
    locks = std::move(o.locks);
    building_threaded = o.building_threaded;

    mapping = o.mapping;
    mapping_size = o.mapping_size;
    o.mapping = nullptr;

    if (mapping) {
        // The file did not move, so the pointers into it still stand.
        vectors = o.vectors;
        levels = o.levels;
        layer0_p = o.layer0_p;
        upper_p = o.upper_p;
        offset_p = o.offset_p;
    } else {
        refresh_pointers();
    }
    return *this;
}

void Hnsw::refresh_pointers() {
    if (mapping) return;
    vectors = vector_store.data();
    levels = node_level.data();
    layer0_p = layer0.data();
    upper_p = upper.data();
    offset_p = link_offset.data();
}

float Hnsw::distance(const float* query, uint32_t node) const {
    return params.simd ? l2_simd(query, vec(node), dim) : l2(query, vec(node), dim);
}

float Hnsw::distance(uint32_t a, uint32_t b) const {
    return params.simd ? l2_simd(vec(a), vec(b), dim) : l2(vec(a), vec(b), dim);
}

uint32_t* Hnsw::links(uint32_t node, int level) {
    if (level == 0) return layer0_p + (size_t)node * (m0 + 1);
    return upper_p + offset_p[node] + (size_t)(level - 1) * (params.M + 1);
}

const uint32_t* Hnsw::links(uint32_t node, int level) const {
    if (level == 0) return layer0_p + (size_t)node * (m0 + 1);
    return upper_p + offset_p[node] + (size_t)(level - 1) * (params.M + 1);
}

// Copied under the lock during a build so the distance work runs unlocked.
const uint32_t* Hnsw::read_links(uint32_t node, int level, uint32_t& count) const {
    const uint32_t* src = links(node, level);

    if (!building_threaded) {
        count = src[0];
        return src + 1;
    }

    if (neighbour_copy.size() < m0 + 1) neighbour_copy.resize(m0 + 1);

    std::shared_lock<std::shared_mutex> guard(locks->node[node % kLockCount]);
    count = std::min<uint32_t>(src[0], max_links(level));
    memcpy(neighbour_copy.data(), src + 1, count * sizeof(uint32_t));
    return neighbour_copy.data();
}

// Expands the nearest candidate until nothing left can beat the worst result held.
void Hnsw::search_layer(const float* query, uint32_t entry, float entry_dist, size_t ef, int level,
                        std::vector<Hit>& found) const {
    candidates.clear();
    found.clear();

    uint32_t stamp = visited.begin(n);
    visited.stamp[entry] = stamp;

    candidates.push_back({entry_dist, entry});
    found.push_back({entry_dist, entry});
    float worst = entry_dist;

    while (!candidates.empty()) {
        std::pop_heap(candidates.begin(), candidates.end(), Nearest{});
        Hit current = candidates.back();
        candidates.pop_back();

        if (current.first > worst && found.size() >= ef) break;

        uint32_t count = 0;
        const uint32_t* neighbours = read_links(current.second, level, count);

        for (uint32_t i = 0; i < count; i++) {
            uint32_t node = neighbours[i];
            if (visited.stamp[node] == stamp) continue;
            visited.stamp[node] = stamp;

            float d = distance(query, node);
            if (found.size() >= ef && d >= worst) continue;

            candidates.push_back({d, node});
            std::push_heap(candidates.begin(), candidates.end(), Nearest{});

            found.push_back({d, node});
            std::push_heap(found.begin(), found.end(), Furthest{});
            if (found.size() > ef) {
                std::pop_heap(found.begin(), found.end(), Furthest{});
                found.pop_back();
            }
            worst = found.front().first;
        }
    }
}

// Upper layers only need the single nearest node, so a plain walk is enough.
uint32_t Hnsw::descend(const float* query, uint32_t entry, float& dist, int level) const {
    bool moved = true;
    while (moved) {
        moved = false;

        uint32_t count = 0;
        const uint32_t* neighbours = read_links(entry, level, count);

        for (uint32_t i = 0; i < count; i++) {
            float d = distance(query, neighbours[i]);
            if (d < dist) {
                dist = d;
                entry = neighbours[i];
                moved = true;
            }
        }
    }
    return entry;
}

// Drops candidates nearer a chosen neighbour than the node, keeping links spread out.
void Hnsw::select_neighbours(std::vector<Hit>& cand, size_t want,
                             std::vector<uint32_t>& out) const {
    out.clear();
    std::sort(cand.begin(), cand.end());

    if (!params.use_heuristic || cand.size() <= want) {
        for (size_t i = 0; i < std::min(want, cand.size()); i++) out.push_back(cand[i].second);
        return;
    }

    std::vector<Hit> discarded;
    for (const Hit& c : cand) {
        if (out.size() >= want) break;

        bool keep = true;
        for (uint32_t picked : out) {
            if (distance(picked, c.second) < c.first) {
                keep = false;
                break;
            }
        }

        if (keep)
            out.push_back(c.second);
        else
            discarded.push_back(c);
    }

    // Too few links is worse than imperfect ones, so put some rejects back.
    for (const Hit& d : discarded) {
        if (out.size() >= want) break;
        out.push_back(d.second);
    }
}

void Hnsw::connect(uint32_t node, const std::vector<uint32_t>& neighbours, int level) {
    size_t cap = max_links(level);

    {
        std::unique_lock<std::shared_mutex> guard;
        if (building_threaded)
            guard = std::unique_lock<std::shared_mutex>(locks->node[node % kLockCount]);

        uint32_t* mine = links(node, level);
        mine[0] = std::min(neighbours.size(), cap);
        for (uint32_t i = 0; i < mine[0]; i++) mine[i + 1] = neighbours[i];
    }

    // Links have to go both ways, or the new node cannot be reached from them.
    for (uint32_t other : neighbours) {
        std::unique_lock<std::shared_mutex> guard;
        if (building_threaded)
            guard = std::unique_lock<std::shared_mutex>(locks->node[other % kLockCount]);

        uint32_t* theirs = links(other, level);
        uint32_t count = theirs[0];

        bool already = false;
        for (uint32_t i = 1; i <= count; i++)
            if (theirs[i] == node) already = true;
        if (already) continue;

        if (count < cap) {
            theirs[count + 1] = node;
            theirs[0] = count + 1;
            continue;
        }

        // Full, so run the same pick over its links plus the new node.
        std::vector<Hit> cand;
        cand.reserve(count + 1);
        for (uint32_t i = 1; i <= count; i++)
            cand.push_back({distance(other, theirs[i]), theirs[i]});
        cand.push_back({distance(other, node), node});

        std::vector<uint32_t> kept;
        select_neighbours(cand, cap, kept);

        theirs[0] = kept.size();
        for (size_t i = 0; i < kept.size(); i++) theirs[i + 1] = kept[i];
    }
}

// Walks down the layers above this node, then links it from its level to zero.
void Hnsw::insert(uint32_t id, const float* v, int level) {
    std::unique_lock<std::mutex> global_guard;
    int top = max_level;

    if (building_threaded) {
        global_guard = std::unique_lock<std::mutex>(locks->global);
        top = max_level;
        if (level <= top) global_guard.unlock();  // only a new top layer needs it held
    }

    if (top < 0) {
        entry_point = id;
        max_level = level;
        return;
    }

    uint32_t entry = entry_point;
    float dist = distance(v, entry);

    for (int lc = top; lc > level; lc--) entry = descend(v, entry, dist, lc);

    for (int lc = std::min(top, level); lc >= 0; lc--) {
        search_layer(v, entry, dist, params.ef_construction, lc, found_scratch);
        select_neighbours(found_scratch, params.M, chosen);
        connect(id, chosen, lc);

        auto best = std::min_element(found_scratch.begin(), found_scratch.end());
        entry = best->second;
        dist = best->first;
    }

    if (level > max_level) {
        entry_point = id;
        max_level = level;
    }
}

// Levels are drawn up front so the graph does not depend on the thread count.
std::vector<int> Hnsw::prepare(const Dataset& ds) {
    if (ds.dim != dim) throw std::invalid_argument("wrong dimension");

    size_t count = ds.size();
    Rng rng(params.seed);

    // floor(-ln(U) / ln(M)), so each layer up holds about one node in M.
    std::vector<int> level(count);
    size_t upper_total = 0;
    for (size_t i = 0; i < count; i++) {
        double u = rng.uniform();
        if (u <= 0) u = 1e-9;
        level[i] = params.hierarchical ? (int)(-std::log(u) * level_scale) : 0;
        upper_total += (size_t)level[i] * (params.M + 1);
    }

    vector_store = ds.data;
    node_level.assign(level.begin(), level.end());
    layer0.assign(count * (m0 + 1), 0);
    upper.assign(upper_total, 0);

    link_offset.assign(count + 1, 0);
    uint64_t running = 0;
    for (size_t i = 0; i < count; i++) {
        link_offset[i] = running;
        running += (size_t)level[i] * (params.M + 1);
    }
    link_offset[count] = running;

    refresh_pointers();
    n = count;
    return level;
}

void Hnsw::build(const Dataset& ds) {
    std::vector<int> level = prepare(ds);
    for (size_t i = 0; i < level.size(); i++) insert(i, ds.at(i), level[i]);
}

void Hnsw::build_threaded(const Dataset& ds, unsigned threads) {
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (threads < 2 || ds.size() < 1000) return build(ds);

    std::vector<int> level = prepare(ds);
    size_t count = level.size();

    insert(0, ds.at(0), level[0]);  // gives the others an entry point to start from

    locks = std::make_unique<Locks>();
    building_threaded = true;

    std::atomic<size_t> next{1};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; t++) {
        pool.emplace_back([&] {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= count) break;
                insert(i, ds.at(i), level[i]);
            }
        });
    }
    for (std::thread& t : pool) t.join();

    building_threaded = false;
    locks.reset();
}

// Algorithm 5. Walk down the upper layers, then one real search at layer 0.
std::vector<Hnsw::Hit> Hnsw::search(const float* query, size_t k, size_t ef) const {
    std::vector<Hit> found;
    if (n == 0) return found;

    uint32_t entry = entry_point;
    float dist = distance(query, entry);
    for (int lc = max_level; lc > 0; lc--) entry = descend(query, entry, dist, lc);

    search_layer(query, entry, dist, std::max(ef, k), 0, found);

    std::sort(found.begin(), found.end());
    if (found.size() > k) found.resize(k);
    return found;
}

std::vector<Hnsw::Hit> Hnsw::search(const float* query, size_t k) const {
    return search(query, k, params.ef_search);
}

size_t Hnsw::memory_bytes() const {
    if (mapping) return mapping_size;
    return vector_store.size() * 4 + layer0.size() * 4 + upper.size() * 4 +
           link_offset.size() * 8 + node_level.size() * 4;
}

std::vector<uint32_t> Hnsw::neighbours(uint32_t node, int level) const {
    const uint32_t* list = links(node, level);
    return std::vector<uint32_t>(list + 1, list + 1 + list[0]);
}

float Hnsw::mean_degree() const {
    if (n == 0) return 0;
    size_t total = 0;
    for (uint32_t i = 0; i < n; i++) total += links(i, 0)[0];
    return (float)total / n;
}

// Sections sit at aligned offsets in the header so load_mapped needs no parsing.
void Hnsw::save(const std::string& path) const {
    Header h{};
    h.magic = kMagic;
    h.dim = dim;
    h.params = params;
    h.n = n;
    h.entry = entry_point;
    h.max_level = max_level;
    h.upper_len = offset_p[n];

    uint64_t at = round_up(sizeof(Header));
    h.off_vectors = at;
    at = round_up(at + (uint64_t)n * dim * 4);
    h.off_levels = at;
    at = round_up(at + (uint64_t)n * 4);
    h.off_layer0 = at;
    at = round_up(at + (uint64_t)n * (m0 + 1) * 4);
    h.off_offsets = at;
    at = round_up(at + ((uint64_t)n + 1) * 8);
    h.off_upper = at;
    at = round_up(at + h.upper_len * 4);
    h.total = at;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);

    // Write the gaps out rather than seeking, which would leave a sparse hole.
    std::vector<char> padding(64, 0);
    auto pad_to = [&](uint64_t target) {
        uint64_t here = f.tellp();
        if (target > here) f.write(padding.data(), target - here);
    };

    f.write((const char*)&h, sizeof(h));

    pad_to(h.off_vectors);
    f.write((const char*)vectors, (uint64_t)n * dim * 4);

    pad_to(h.off_levels);
    f.write((const char*)levels, (uint64_t)n * 4);

    pad_to(h.off_layer0);
    f.write((const char*)layer0_p, (uint64_t)n * (m0 + 1) * 4);

    pad_to(h.off_offsets);
    f.write((const char*)offset_p, ((uint64_t)n + 1) * 8);

    pad_to(h.off_upper);
    if (h.upper_len) f.write((const char*)upper_p, h.upper_len * 4);

    pad_to(h.total);
}

Hnsw Hnsw::load_mapped(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);

    struct stat st {};
    fstat(fd, &st);
    void* base = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);

    const Header* h = (const Header*)base;
    if (h->magic != kMagic) {
        munmap(base, st.st_size);
        throw std::runtime_error("not a vecsearch index: " + path);
    }

    Hnsw idx(h->dim, h->params);
    idx.mapping = base;
    idx.mapping_size = st.st_size;
    idx.n = h->n;
    idx.entry_point = h->entry;
    idx.max_level = h->max_level;

    char* p = (char*)base;
    idx.vectors = (const float*)(p + h->off_vectors);
    idx.levels = (const int32_t*)(p + h->off_levels);
    idx.layer0_p = (uint32_t*)(p + h->off_layer0);
    idx.offset_p = (const uint64_t*)(p + h->off_offsets);
    idx.upper_p = (uint32_t*)(p + h->off_upper);
    return idx;
}

Hnsw Hnsw::load(const std::string& path) {
    Hnsw mapped = load_mapped(path);
    uint64_t upper_len = mapped.offset_p[mapped.n];

    Hnsw out(mapped.dim, mapped.params);
    out.n = mapped.n;
    out.entry_point = mapped.entry_point;
    out.max_level = mapped.max_level;

    out.vector_store.assign(mapped.vectors, mapped.vectors + (size_t)mapped.n * mapped.dim);
    out.node_level.assign(mapped.levels, mapped.levels + mapped.n);
    out.layer0.assign(mapped.layer0_p, mapped.layer0_p + (size_t)mapped.n * (mapped.m0 + 1));
    out.link_offset.assign(mapped.offset_p, mapped.offset_p + mapped.n + 1);
    out.upper.assign(mapped.upper_p, mapped.upper_p + upper_len);
    out.refresh_pointers();
    return out;
}
