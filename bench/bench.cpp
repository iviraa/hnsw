// Recall against throughput as the search width varies. Usage: ./bench data/sift sift
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "dataset.h"
#include "eval.h"
#include "hnsw.h"

using clock_type = std::chrono::steady_clock;
static double seconds_since(clock_type::time_point t) {
    return std::chrono::duration<double>(clock_type::now() - t).count();
}

int main(int argc, char** argv) {
    const size_t k = 10;
    Dataset base, queries;
    std::vector<std::vector<uint32_t>> truth;

    if (argc >= 3) {
        std::string dir = argv[1], stem = argv[2];
        base = read_fvecs(dir + "/" + stem + "_base.fvecs");
        queries = read_fvecs(dir + "/" + stem + "_query.fvecs");
        truth = read_ivecs(dir + "/" + stem + "_groundtruth.ivecs");
        for (auto& t : truth)
            if (t.size() > k) t.resize(k);
        if (queries.size() > 1000) {
            queries.data.resize(1000 * queries.dim);
            truth.resize(1000);
        }
    } else {
        base = make_test_data(50000, 128, 100, 1);
        queries = make_test_data(500, 128, 100, 2);
        printf("computing exact answers for the test data ...\n");
        truth = exact_neighbours(queries, base, k);
    }

    printf("\n%zu vectors, dim %zu, %zu queries, k=%zu\n\n", base.size(), base.dim, queries.size(), k);

    // Exact search baseline, limited to a few queries as it is slow.
    size_t probe = std::min<size_t>(queries.size(), 20);
    auto t0 = clock_type::now();
    for (size_t i = 0; i < probe; i++) brute_force(queries.at(i), base, k);
    double exact = seconds_since(t0);
    printf("exact search: %.0f queries/sec, %.2f ms each\n\n", probe / exact, exact * 1000 / probe);

    Params p;
    Hnsw index(base.dim, p);
    t0 = clock_type::now();
    index.build_threaded(base, 0);
    printf("built in %.1fs, mean degree %.1f, %d layers, %.0f MB\n\n", seconds_since(t0),
           index.mean_degree(), index.top_level() + 1, index.memory_bytes() / 1048576.0);

    printf("%8s %10s %12s %12s\n", "ef", "recall", "queries/sec", "us/query");
    for (size_t ef : {16, 32, 48, 64, 96, 128, 200, 320}) {
        float total = 0;
        for (size_t i = 0; i < queries.size(); i++) {
            auto hits = index.search(queries.at(i), k, ef);
            std::vector<uint32_t> ids;
            for (auto& h : hits) ids.push_back(h.second);
            total += recall(ids, truth[i], k);
        }

        // Best of three runs.
        double best = 1e30;
        for (int run = 0; run < 3; run++) {
            auto start = clock_type::now();
            for (size_t i = 0; i < queries.size(); i++) index.search(queries.at(i), k, ef);
            double elapsed = seconds_since(start);
            if (elapsed < best) best = elapsed;
        }

        printf("%8zu %10.4f %12.0f %12.1f\n", ef, total / queries.size(), queries.size() / best,
               best * 1e6 / queries.size());
    }
    printf("\n");
    return 0;
}
