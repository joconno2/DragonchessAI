#pragma once

#include "td_features.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <array>

namespace dragonchess {

// NNUE-style network: input(4060) → hidden1(512,ClippedReLU) → hidden2(64,ClippedReLU) → output(1)
//
// The first layer uses an incremental accumulator: instead of recomputing
// h1 = W1 * sparse_input + b1 at every leaf node, the accumulator is
// updated incrementally when pieces move. Only layers 2-3 are computed
// fresh at each evaluation.
struct NNWeights {
    static constexpr int N_INPUT = NUM_TD_FEATURES;  // 32284 (8 king buckets x 14 pieces x 288 squares + 28 strategic)
    static constexpr int N_HIDDEN1 = 512;
    static constexpr int N_HIDDEN2 = 64;
    static constexpr int N_PSQ = NUM_PIECE_SQUARE;    // 32256 (king-bucket piece-square features)

    // Layer 1: N_INPUT → N_HIDDEN1 (row-major: w1[h * N_INPUT + i])
    std::vector<float> w1;
    std::vector<float> b1;

    // Layer 2: N_HIDDEN1 → N_HIDDEN2
    std::vector<float> w2;
    std::vector<float> b2;

    // Layer 3: N_HIDDEN2 → 1
    std::vector<float> w3;
    float b3;

    NNWeights() : b3(0.0f) {
        w1.resize(N_HIDDEN1 * N_INPUT, 0.0f);
        b1.resize(N_HIDDEN1, 0.0f);
        w2.resize(N_HIDDEN2 * N_HIDDEN1, 0.0f);
        b2.resize(N_HIDDEN2, 0.0f);
        w3.resize(N_HIDDEN2, 0.0f);
    }

    // Full forward pass from sparse features (used for initial computation).
    float forward(const std::vector<SparseFeature>& input) const {
        float h1[N_HIDDEN1];
        compute_h1(input, h1);
        return forward_from_h1(h1);
    }

    // Compute first hidden layer from sparse input into h1 array.
    void compute_h1(const std::vector<SparseFeature>& input, float* h1) const {
        for (int j = 0; j < N_HIDDEN1; ++j)
            h1[j] = b1[j];
        for (const auto& sf : input)
            for (int j = 0; j < N_HIDDEN1; ++j)
                h1[j] += w1[j * N_INPUT + sf.index] * sf.value;
    }

    // Forward pass from pre-computed h1 (layers 2-3 only). Fast.
    float forward_from_h1(const float* h1) const {
        // ReLU on h1
        float a1[N_HIDDEN1];
        for (int j = 0; j < N_HIDDEN1; ++j)
            a1[j] = std::max(h1[j], 0.0f);

        // Layer 2
        float h2[N_HIDDEN2];
        for (int k = 0; k < N_HIDDEN2; ++k) {
            float sum = b2[k];
            for (int j = 0; j < N_HIDDEN1; ++j)
                sum += w2[k * N_HIDDEN1 + j] * a1[j];
            h2[k] = std::max(sum, 0.0f);
        }

        // Layer 3
        float out = b3;
        for (int k = 0; k < N_HIDDEN2; ++k)
            out += w3[k] * h2[k];
        return out;
    }

    // --- Incremental accumulator helpers ---

    // Add a piece-square feature to the accumulator.
    // feat_idx is the piece-square index (0..4031), sign is +1 or -1.
    void acc_add_feature(float* h1, int feat_idx, float value) const {
        for (int j = 0; j < N_HIDDEN1; ++j)
            h1[j] += w1[j * N_INPUT + feat_idx] * value;
    }

    // Remove a piece-square feature from the accumulator.
    void acc_remove_feature(float* h1, int feat_idx, float value) const {
        for (int j = 0; j < N_HIDDEN1; ++j)
            h1[j] -= w1[j * N_INPUT + feat_idx] * value;
    }

    static constexpr int total_params() {
        return N_HIDDEN1 * N_INPUT + N_HIDDEN1
             + N_HIDDEN2 * N_HIDDEN1 + N_HIDDEN2
             + N_HIDDEN2 + 1;
    }

    std::vector<float> to_flat() const {
        std::vector<float> flat;
        flat.reserve(total_params());
        flat.insert(flat.end(), w1.begin(), w1.end());
        flat.insert(flat.end(), b1.begin(), b1.end());
        flat.insert(flat.end(), w2.begin(), w2.end());
        flat.insert(flat.end(), b2.begin(), b2.end());
        flat.insert(flat.end(), w3.begin(), w3.end());
        flat.push_back(b3);
        return flat;
    }

    bool from_flat(const std::vector<float>& flat) {
        if (static_cast<int>(flat.size()) != total_params())
            return false;
        int offset = 0;
        auto copy_n = [&](std::vector<float>& dst, int n) {
            dst.assign(flat.begin() + offset, flat.begin() + offset + n);
            offset += n;
        };
        copy_n(w1, N_HIDDEN1 * N_INPUT);
        copy_n(b1, N_HIDDEN1);
        copy_n(w2, N_HIDDEN2 * N_HIDDEN1);
        copy_n(b2, N_HIDDEN2);
        copy_n(w3, N_HIDDEN2);
        b3 = flat[offset];
        return true;
    }
};

// Accumulator: pre-computed first hidden layer values.
// Pushed/popped alongside make_move/undo_move in AB search.
struct NNAccumulator {
    std::array<float, NNWeights::N_HIDDEN1> h1;
};

} // namespace dragonchess
