// Dumps the graph and one search step by step as JSON. Usage: ./trace > graph.json
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

#include "distance.h"
#include "plane.h"

using Hit = std::pair<float, uint32_t>;



static void print_ids(const std::vector<uint32_t>& v) {
    printf("[");
    for (size_t i = 0; i < v.size(); i++) printf(i ? ",%u" : "%u", v[i]);
    printf("]");
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? atoi(argv[1]) : 260;
    size_t ef = argc > 2 ? atoi(argv[2]) : 16;
    const size_t k = 5;

    Dataset ds = spread_points(n, 7);
    Hnsw index(2, demo_params());
    index.build(ds);

    float lo = ds.data[0], hi = ds.data[0];
    for (float v : ds.data) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    std::vector<float> query{lo + (hi - lo) * 0.72f, lo + (hi - lo) * 0.30f};
    auto dist = [&](uint32_t id) { return l2(query.data(), ds.at(id), 2); };

    printf("{\n");
    printf("\"query\":[%.6f,%.6f],\n", query[0], query[1]);
    printf("\"top\":%d,\"ef\":%zu,\"k\":%zu,\n", index.top_level(), ef, k);

    printf("\"points\":[");
    for (uint32_t i = 0; i < index.size(); i++)
        printf(i ? ",[%.6f,%.6f]" : "[%.6f,%.6f]", ds.at(i)[0], ds.at(i)[1]);
    printf("],\n");

    printf("\"levels\":[");
    for (uint32_t i = 0; i < index.size(); i++) printf(i ? ",%d" : "%d", index.level_of(i));
    printf("],\n");

    printf("\"layers\":[\n");
    for (int level = index.top_level(); level >= 0; level--) {
        printf("%s{\"level\":%d,\"edges\":[", level == index.top_level() ? "" : ",\n", level);
        bool first = true;
        for (uint32_t i = 0; i < index.size(); i++) {
            if (index.level_of(i) < level) continue;
            for (uint32_t j : index.neighbours(i, level)) {
                if (j <= i) continue;
                printf(first ? "[%u,%u]" : ",[%u,%u]", i, j);
                first = false;
            }
        }
        printf("]}");
    }
    printf("\n],\n");

    // The search is replayed through the accessors, so the index needs no logging.
    printf("\"steps\":[\n");
    bool first_step = true;
    auto step = [&](int level, const char* kind, uint32_t at, const std::vector<uint32_t>& path,
                    const std::vector<uint32_t>& fresh, const std::vector<uint32_t>& result) {
        printf("%s{\"level\":%d,\"kind\":\"%s\",\"at\":%u,\"path\":", first_step ? "" : ",\n", level,
               kind, at);
        print_ids(path);
        printf(",\"fresh\":");
        print_ids(fresh);
        printf(",\"result\":");
        print_ids(result);
        printf("}");
        first_step = false;
    };

    uint32_t entry = index.entry();
    for (int level = index.top_level(); level > 0; level--) {
        std::vector<uint32_t> path{entry};
        step(level, "start", entry, path, {entry}, {});

        float best = dist(entry);
        bool moved = true;
        while (moved) {
            moved = false;
            std::vector<uint32_t> looked;
            for (uint32_t nb : index.neighbours(entry, level)) {
                looked.push_back(nb);
                float d = dist(nb);
                if (d < best) {
                    best = d;
                    entry = nb;
                    moved = true;
                }
            }
            if (moved) {
                path.push_back(entry);
                step(level, "hop", entry, path, looked, {});
            } else {
                step(level, "settle", entry, path, looked, {});
            }
        }
    }

    {
        std::vector<bool> seen(index.size(), false);
        std::priority_queue<Hit, std::vector<Hit>, std::greater<Hit>> cand;
        std::priority_queue<Hit> result;
        std::vector<uint32_t> path{entry};

        float d0 = dist(entry);
        cand.push({d0, entry});
        result.push({d0, entry});
        seen[entry] = true;
        step(0, "start", entry, path, {entry}, {entry});

        while (!cand.empty()) {
            Hit near = cand.top();
            if (near.first > result.top().first && result.size() >= ef) break;
            cand.pop();

            std::vector<uint32_t> fresh;
            for (uint32_t nb : index.neighbours(near.second, 0)) {
                if (seen[nb]) continue;
                seen[nb] = true;
                fresh.push_back(nb);
                float d = dist(nb);
                if (result.size() < ef || d < result.top().first) {
                    cand.push({d, nb});
                    result.push({d, nb});
                    if (result.size() > ef) result.pop();
                }
            }

            std::vector<Hit> held;
            auto copy = result;
            while (!copy.empty()) {
                held.push_back(copy.top());
                copy.pop();
            }
            std::sort(held.begin(), held.end());
            std::vector<uint32_t> ids;
            for (const Hit& h : held) ids.push_back(h.second);

            path.push_back(near.second);
            step(0, "expand", near.second, path, fresh, ids);
        }

        std::vector<Hit> held;
        while (!result.empty()) {
            held.push_back(result.top());
            result.pop();
        }
        std::sort(held.begin(), held.end());
        held.resize(std::min(k, held.size()));

        printf("\n],\n\"answer\":[");
        for (size_t i = 0; i < held.size(); i++) printf(i ? ",%u" : "%u", held[i].second);
        printf("],\n");
    }

    std::vector<Hit> all;
    for (uint32_t i = 0; i < index.size(); i++) all.push_back({dist(i), i});
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    printf("\"exact\":[");
    for (size_t i = 0; i < k; i++) printf(i ? ",%u" : "%u", all[i].second);
    printf("]\n}\n");
    return 0;
}
