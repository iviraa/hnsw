// Covers what bench cannot see: saving, mapping, and the threaded build.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dataset.h"
#include "eval.h"
#include "hnsw.h"

static int failures = 0;

static void check(const char* what, bool ok) {
    printf("  %s  %s\n", ok ? "pass" : "FAIL", what);
    if (!ok) failures++;
}

static float measure(const Hnsw& index, const Dataset& queries,
                     const std::vector<std::vector<uint32_t>>& truth) {
    float total = 0;
    for (size_t i = 0; i < queries.size(); i++) {
        std::vector<uint32_t> ids;
        for (auto& h : index.search(queries.at(i), 10, 64)) ids.push_back(h.second);
        total += recall(ids, truth[i], 10);
    }
    return total / queries.size();
}

// A stored vector has to find itself, which catches most indexing errors.
static void test_sanity() {
    printf("\nindex\n");
    Dataset ds = make_test_data(5000, 32, 20, 1);
    Hnsw index(ds.dim, Params{});
    index.build(ds);

    int found = 0;
    for (size_t i = 0; i < 5000; i += 97) {
        auto hits = index.search(ds.at(i), 1, 64);
        if (!hits.empty() && hits[0].second == i) found++;
    }
    check("every probed vector finds itself", found == 52);
    check("nobody is left with no links", index.mean_degree() > 1.0f);
}

// The graphs differ with scheduling, so only recall can be compared.
static void test_threaded_build() {
    printf("\nthreaded build\n");
    Dataset base = make_test_data(20000, 32, 30, 3);
    Dataset queries = make_test_data(200, 32, 30, 77);
    auto truth = exact_neighbours(queries, base, 10);

    Hnsw serial(base.dim, Params{});
    serial.build(base);

    Hnsw parallel(base.dim, Params{});
    parallel.build_threaded(base, 0);

    float a = measure(serial, queries, truth);
    float b = measure(parallel, queries, truth);
    printf("        serial %.3f   threaded %.3f\n", a, b);

    check("threaded build holds every vector", parallel.size() == base.size());
    check("threaded recall stays close to serial", b > a - 0.03f);
}

// The graph is unchanged, only the storage, so answers must match exactly.
static void test_save_and_map() {
    printf("\nsave and load\n");
    Dataset ds = make_test_data(5000, 32, 20, 4);
    Hnsw index(ds.dim, Params{});
    index.build(ds);

    const char* path = "/tmp/hnsw_test.idx";
    index.save(path);

    Hnsw copied = Hnsw::load(path);
    Hnsw mapped = Hnsw::load_mapped(path);
    check("loaded index has the same size", copied.size() == index.size());
    check("mapped index has the same size", mapped.size() == index.size());

    int same_copied = 0, same_mapped = 0;
    for (size_t i = 0; i < 200; i++) {
        auto want = index.search(ds.at(i * 7), 10, 64);
        auto got_c = copied.search(ds.at(i * 7), 10, 64);
        auto got_m = mapped.search(ds.at(i * 7), 10, 64);
        if (want.size() == got_c.size() && want[0].second == got_c[0].second) same_copied++;
        if (want.size() == got_m.size() && want[0].second == got_m[0].second) same_mapped++;
    }
    check("loaded index gives the same answers", same_copied == 200);
    check("mapped index gives the same answers", same_mapped == 200);
    remove(path);
}

int main() {
    printf("\nhnsw tests\n");
    test_sanity();
    test_threaded_build();
    test_save_and_map();
    printf("\n%s\n\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
