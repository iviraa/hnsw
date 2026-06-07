// Draws the layers as an SVG with one query's path traced. Usage: ./draw > graph.svg
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dataset.h"
#include "distance.h"
#include "hnsw.h"

static const int kPanel = 300;    // pixels per layer panel
static const int kGap = 26;
static const int kPad = 34;

struct Point {
    float x, y;
};

// Spread evenly, since the clustered generator bunches everything into blobs.
static Dataset spread_points(size_t n, uint64_t seed) {
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

static Params demo_params() {
    Params p;
    p.M = 6;  // small so the edges stay readable
    p.ef_construction = 60;
    return p;
}



// Maps a vector into panel coordinates.
static Point place(const float* v, float lo, float hi, int panel_x) {
    float span = hi - lo;
    float px = (v[0] - lo) / span * (kPanel - 2 * kPad) + kPad;
    float py = (v[1] - lo) / span * (kPanel - 2 * kPad) + kPad;
    return {panel_x + px, kPanel - py};  // svg y grows downward
}

// Repeats the greedy walk the index performs, recording where it goes.
static std::vector<uint32_t> walk(const Hnsw& index, const Dataset& ds, const float* query,
                                  int level, uint32_t start) {
    std::vector<uint32_t> path{start};
    uint32_t at = start;
    float best = l2(query, ds.at(at), ds.dim);

    bool moved = true;
    while (moved) {
        moved = false;
        for (uint32_t n : index.neighbours(at, level)) {
            float d = l2(query, ds.at(n), ds.dim);
            if (d < best) {
                best = d;
                at = n;
                moved = true;
            }
        }
        if (moved) path.push_back(at);
    }
    return path;
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? atoi(argv[1]) : 260;

    Dataset ds = spread_points(n, 7);
    Hnsw index(2, demo_params());
    index.build(ds);

    float lo = ds.data[0], hi = ds.data[0];
    for (float v : ds.data) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    int layers = index.top_level() + 1;
    std::vector<float> query{lo + (hi - lo) * 0.72f, lo + (hi - lo) * 0.30f};

    int width = layers * kPanel + (layers - 1) * kGap;
    printf("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' "
           "font-family='ui-monospace,SFMono-Regular,Menlo,monospace' font-size='11'>\n",
           width, kPanel + 34);
    printf("<rect width='100%%' height='100%%' fill='#fff'/>\n");

    // Highest layer on the left, which is the order a search visits them.
    uint32_t entry = index.entry();
    for (int level = index.top_level(); level >= 0; level--) {
        int slot = index.top_level() - level;
        int ox = slot * (kPanel + kGap);

        printf("<rect x='%d' y='0' width='%d' height='%d' fill='none' stroke='#e5e5e5'/>\n", ox,
               kPanel, kPanel);

        size_t members = 0;
        for (uint32_t i = 0; i < index.size(); i++)
            if (index.level_of(i) >= level) members++;
        const char* note = level == index.top_level() ? "entry"
                           : level == 0                ? "full set"
                                                       : "";
        printf("<text x='%d' y='%d' fill='#666'>layer %d &#183; %zu nodes %s</text>\n", ox + 4,
               kPanel + 16, level, members, note);

        // Edges first so nodes sit on top of them.
        for (uint32_t i = 0; i < index.size(); i++) {
            if (index.level_of(i) < level) continue;
            Point a = place(ds.at(i), lo, hi, ox);
            for (uint32_t j : index.neighbours(i, level)) {
                if (j <= i) continue;  // draw each edge once
                Point b = place(ds.at(j), lo, hi, ox);
                printf("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#d4d4d4' "
                       "stroke-width='%.2f'/>\n",
                       a.x, a.y, b.x, b.y, level == 0 ? 0.4 : 1.0);
            }
        }

        for (uint32_t i = 0; i < index.size(); i++) {
            if (index.level_of(i) < level) continue;
            Point a = place(ds.at(i), lo, hi, ox);
            printf("<circle cx='%.1f' cy='%.1f' r='2' fill='#999'/>\n", a.x, a.y);
        }

        // The path this layer contributes, and the node handed to the next.
        std::vector<uint32_t> path = walk(index, ds, query.data(), level, entry);
        for (size_t i = 1; i < path.size(); i++) {
            Point a = place(ds.at(path[i - 1]), lo, hi, ox);
            Point b = place(ds.at(path[i]), lo, hi, ox);
            printf("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#111' "
                   "stroke-width='1.6'/>\n",
                   a.x, a.y, b.x, b.y);
        }
        for (uint32_t node : path) {
            Point a = place(ds.at(node), lo, hi, ox);
            printf("<circle cx='%.1f' cy='%.1f' r='3.2' fill='#111'/>\n", a.x, a.y);
        }

        Point q = place(query.data(), lo, hi, ox);
        printf("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#b91c1c' "
               "stroke-width='1.4'/>\n",
               q.x - 5, q.y - 5, q.x + 5, q.y + 5);
        printf("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#b91c1c' "
               "stroke-width='1.4'/>\n",
               q.x - 5, q.y + 5, q.x + 5, q.y - 5);

        entry = path.back();  // the next layer starts where this one stopped
    }

    printf("</svg>\n");
    return 0;
}
