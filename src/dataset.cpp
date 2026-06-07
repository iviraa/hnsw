#include "dataset.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(n);
    f.read(buf.data(), n);
    return buf;
}

// memcpy rather than a cast, which would break strict aliasing.
static int read_int(const char* p) {
    int v;
    memcpy(&v, p, 4);
    return v;
}

static float read_float(const char* p) {
    float v;
    memcpy(&v, p, 4);
    return v;
}

Dataset read_fvecs(const std::string& path) {
    std::vector<char> buf = read_file(path);
    if (buf.size() < 4) throw std::runtime_error("file too short: " + path);

    size_t dim = read_int(buf.data());
    size_t record = 4 + dim * 4;
    if (buf.size() % record != 0) throw std::runtime_error("bad file: " + path);
    size_t n = buf.size() / record;

    Dataset ds;
    ds.dim = dim;
    ds.data.resize(n * dim);
    for (size_t i = 0; i < n; i++) {
        const char* p = buf.data() + i * record + 4;
        for (size_t j = 0; j < dim; j++) ds.data[i * dim + j] = read_float(p + j * 4);
    }
    return ds;
}

std::vector<std::vector<uint32_t>> read_ivecs(const std::string& path) {
    std::vector<char> buf = read_file(path);
    std::vector<std::vector<uint32_t>> out;

    size_t pos = 0;
    while (pos + 4 <= buf.size()) {
        size_t dim = read_int(buf.data() + pos);
        pos += 4;
        if (pos + dim * 4 > buf.size()) break;

        std::vector<uint32_t> row(dim);
        for (size_t j = 0; j < dim; j++) row[j] = read_int(buf.data() + pos + j * 4);
        pos += dim * 4;
        out.push_back(row);
    }
    return out;
}

// Small LCG so the test data is reproducible without pulling in a dependency.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed * 2 + 1) {}

    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }
    float uniform() { return (next() >> 40) / 16777216.0f; }
    // Twelve uniforms summed lands close enough to a normal.
    float normal() {
        float s = 0;
        for (int i = 0; i < 12; i++) s += uniform();
        return s - 6.0f;
    }
};

Dataset make_test_data(size_t n, size_t dim, size_t clusters, uint64_t seed) {
    Rng centre_rng(0xC0FFEE);
    std::vector<float> centres(clusters * dim);
    for (float& c : centres) c = centre_rng.normal() * 10.0f;

    Rng rng(seed);
    Dataset ds;
    ds.dim = dim;
    ds.data.resize(n * dim);
    for (size_t i = 0; i < n; i++) {
        size_t c = rng.next() % clusters;
        for (size_t j = 0; j < dim; j++)
            ds.data[i * dim + j] = centres[c * dim + j] + rng.normal();
    }
    return ds;
}
