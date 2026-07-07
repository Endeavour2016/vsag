
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "proximity_scorer.h"

#include <cmath>
#include <vector>

#include "unittest.h"

#define REQUIRE_APPROX(a, b) REQUIRE(std::abs((a) - (b)) < 1e-6f)

using namespace vsag;

static std::vector<PosSpan>
to_spans(const std::vector<std::vector<uint16_t>>& lists) {
    std::vector<PosSpan> spans;
    spans.reserve(lists.size());
    for (const auto& list : lists) {
        spans.push_back(PosSpan{list.data(), static_cast<uint32_t>(list.size())});
    }
    return spans;
}

TEST_CASE("ProximityScorer Empty Input", "[ut][ProximityScorer]") {
    // No position lists at all
    std::vector<std::vector<uint16_t>> empty;
    float boost = compute_pairwise_proximity(to_spans(empty), false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("ProximityScorer Single Term", "[ut][ProximityScorer]") {
    // Only one term present — no pairs, boost should be 0
    std::vector<std::vector<uint16_t>> single = {{5, 10, 20}};
    float boost = compute_pairwise_proximity(to_spans(single), false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("ProximityScorer Two Terms Adjacent", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 1 → dist=1 → 1/(1+1) = 0.5
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(boost, 0.5f);
}

TEST_CASE("ProximityScorer Two Terms Same Position", "[ut][ProximityScorer]") {
    // term A at pos 5, term B at pos 5 → dist=0 → 1/(0+1) = 1.0
    std::vector<std::vector<uint16_t>> positions = {{5}, {5}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(boost, 1.0f);
}

TEST_CASE("ProximityScorer Two Terms Far Apart", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 100 → dist=100 → 1/101 ≈ 0.0099
    std::vector<std::vector<uint16_t>> positions = {{0}, {100}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(boost, 1.0f / 101.0f);
}

TEST_CASE("ProximityScorer Two Terms Multiple Positions", "[ut][ProximityScorer]") {
    // term A at [0, 50, 200], term B at [3, 48, 300]
    // min_dist = min(|0-3|, |0-48|, |0-300|, |50-3|, |50-48|, |50-300|, |200-3|, ...) = |50-48| = 2
    // But the algorithm should find min_dist using sorted merge: min over all pairs = 2
    // boost = 1/(2+1) = 0.333...
    std::vector<std::vector<uint16_t>> positions = {{0, 50, 200}, {3, 48, 300}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Three Terms Pairwise", "[ut][ProximityScorer]") {
    // term A at pos 0, term B at pos 1, term C at pos 200
    // pairs: (A,B) dist=1 → 0.5, (A,C) dist=200 → 1/201, (B,C) dist=199 → 1/200
    // total ≈ 0.5 + 0.00498 + 0.005 ≈ 0.51
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}, {200}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    float expected = 1.0f / 2.0f + 1.0f / 201.0f + 1.0f / 200.0f;
    REQUIRE_APPROX(boost, expected);
}

TEST_CASE("ProximityScorer Ordered Forward", "[ut][ProximityScorer]") {
    // ordered=true, term A at pos 0, term B at pos 5
    // A before B in query order, pos_A < pos_B → forward → dist = 5
    // boost = 1/(5+1) = 1/6
    std::vector<std::vector<uint16_t>> positions = {{0}, {5}};
    float boost = compute_pairwise_proximity(to_spans(positions), true);
    REQUIRE_APPROX(boost, 1.0f / 6.0f);
}

TEST_CASE("ProximityScorer Ordered Reverse Penalty", "[ut][ProximityScorer]") {
    // ordered=true, term A at pos 10, term B at pos 5
    // A before B in query order, but pos_A > pos_B → reverse → dist = (10-5)*2 = 10
    // boost = 1/(10+1) = 1/11
    std::vector<std::vector<uint16_t>> positions = {{10}, {5}};
    float boost = compute_pairwise_proximity(to_spans(positions), true);
    REQUIRE_APPROX(boost, 1.0f / 11.0f);
}

TEST_CASE("ProximityScorer Ordered vs Unordered", "[ut][ProximityScorer]") {
    // Same positions, ordered should give lower boost due to penalty
    std::vector<std::vector<uint16_t>> positions = {{10}, {5}};
    float boost_unordered = compute_pairwise_proximity(to_spans(positions), false);
    float boost_ordered = compute_pairwise_proximity(to_spans(positions), true);
    // Unordered: dist=5, boost=1/6
    // Ordered: dist=10 (reverse penalty), boost=1/11
    REQUIRE(boost_unordered > boost_ordered);
    REQUIRE_APPROX(boost_unordered, 1.0f / 6.0f);
    REQUIRE_APPROX(boost_ordered, 1.0f / 11.0f);
}

TEST_CASE("ProximityScorer Multiple Positions Min Distance", "[ut][ProximityScorer]") {
    // term A at [10, 100], term B at [12, 95]
    // Unordered min_dist: min(|10-12|, |10-95|, |100-12|, |100-95|) = min(2,85,88,5) = 2
    // boost = 1/3
    std::vector<std::vector<uint16_t>> positions = {{10, 100}, {12, 95}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Ordered Multiple Positions Best Forward", "[ut][ProximityScorer]") {
    // ordered=true, term A at [10, 100], term B at [12, 95]
    // Forward pairs (pos_A < pos_B): (10,12) dist=2, (10,95) dist=85, (100, -)  none forward
    // Reverse pairs (pos_A > pos_B): (100,12) dist=88*2=176, (100,95) dist=5*2=10
    // Min dist overall: 2 (forward pair 10→12)
    // boost = 1/3
    std::vector<std::vector<uint16_t>> positions = {{10, 100}, {12, 95}};
    float boost = compute_pairwise_proximity(to_spans(positions), true);
    REQUIRE_APPROX(boost, 1.0f / 3.0f);
}

TEST_CASE("ProximityScorer Empty Position List Skipped", "[ut][ProximityScorer]") {
    // term A has positions, term B has empty (term not present in doc)
    // This pair contributes nothing
    std::vector<std::vector<uint16_t>> positions = {{5, 10}, {}};
    float boost = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE(boost == 0.0f);
}

// ===== calculate_pairwise_proximity (adjacent-only) tests =====

TEST_CASE("CalcProximity Adjacent Only Three Terms", "[ut][ProximityScorer]") {
    // A at 0, B at 1, C at 10. Adjacent pairs: (A,B) dist=1 → 0.5,
    // (B,C) dist=9 → 0.1. (A,C) is NOT scored. Total = 0.6.
    // compute_pairwise_proximity would also add (A,C) dist=10 → 1/11 ≈ 0.0909.
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}, {10}};
    float adj = calculate_pairwise_proximity(to_spans(positions), false);
    REQUIRE_APPROX(adj, 0.5f + 0.1f);
    float all = compute_pairwise_proximity(to_spans(positions), false);
    REQUIRE(all > adj);  // all-pairs includes the extra (A,C) contribution
}

TEST_CASE("CalcProximity Adjacent Only Empty Term Skipped", "[ut][ProximityScorer]") {
    // B empty → both (A,B) and (B,C) skipped → boost 0.
    std::vector<std::vector<uint16_t>> positions = {{0}, {}, {2}};
    float boost = calculate_pairwise_proximity(to_spans(positions), false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("CalcProximity Adjacent Only Single Term", "[ut][ProximityScorer]") {
    std::vector<std::vector<uint16_t>> single = {{5, 10}};
    float boost = calculate_pairwise_proximity(to_spans(single), false);
    REQUIRE(boost == 0.0f);
}

TEST_CASE("CalcProximity Adjacent Only Ordered Reverse Penalty", "[ut][ProximityScorer]") {
    // A at 5, B at 2. Ordered reverse: dist = (5-2)*2 = 6 → 1/7.
    std::vector<std::vector<uint16_t>> positions = {{5}, {2}};
    float boost = calculate_pairwise_proximity(to_spans(positions), true);
    REQUIRE_APPROX(boost, 1.0f / 7.0f);
}

// ===== check_phrase_constraint tests =====

TEST_CASE("PhraseFilter All Terms Present Within Slop", "[ut][ProximityScorer][PhraseFilter]") {
    // terms at [0, 1, 2], slop=0 → span=2, max_span = 0+3-1=2 → pass
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}, {2}};
    REQUIRE(check_phrase_constraint(positions, 0, false) == true);
}

TEST_CASE("PhraseFilter Exceeds Slop", "[ut][ProximityScorer][PhraseFilter]") {
    // terms at [0, 1, 100], slop=5 → span=100, max_span=5+3-1=7 → fail
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}, {100}};
    REQUIRE(check_phrase_constraint(positions, 5, false) == false);
}

TEST_CASE("PhraseFilter Missing Term", "[ut][ProximityScorer][PhraseFilter]") {
    // term B has no positions → fail (all terms must be present)
    std::vector<std::vector<uint16_t>> positions = {{0}, {}, {2}};
    REQUIRE(check_phrase_constraint(positions, 100, false) == false);
}

TEST_CASE("PhraseFilter Single Term Always Passes", "[ut][ProximityScorer][PhraseFilter]") {
    // Only 1 term → span=0, always passes
    std::vector<std::vector<uint16_t>> positions = {{5}};
    REQUIRE(check_phrase_constraint(positions, 0, false) == true);
}

TEST_CASE("PhraseFilter Multiple Positions Best Span", "[ut][ProximityScorer][PhraseFilter]") {
    // term A at [0, 50], term B at [3, 48]
    // Best span: [48, 50] = 2, or [0, 3] = 3
    // With slop=2: max_span = 2+2-1 = 3 → span=2 passes
    std::vector<std::vector<uint16_t>> positions = {{0, 50}, {3, 48}};
    REQUIRE(check_phrase_constraint(positions, 2, false) == true);
}

TEST_CASE("PhraseFilter Multiple Positions Fail", "[ut][ProximityScorer][PhraseFilter]") {
    // term A at [0, 50], term B at [10, 60]
    // Best span: min(|0-10|, |50-60|) → [0,10]=10 or [50,60]=10
    // slop=5: max_span = 5+2-1 = 6 → 10 > 6 → fail
    std::vector<std::vector<uint16_t>> positions = {{0, 50}, {10, 60}};
    REQUIRE(check_phrase_constraint(positions, 5, false) == false);
}

TEST_CASE("PhraseFilter Ordered Pass", "[ut][ProximityScorer][PhraseFilter]") {
    // terms A at [0], B at [2] → ordered: A before B → pass
    std::vector<std::vector<uint16_t>> positions = {{0}, {2}};
    REQUIRE(check_phrase_constraint(positions, 5, true) == true);
}

TEST_CASE("PhraseFilter Ordered Fail Reverse", "[ut][ProximityScorer][PhraseFilter]") {
    // terms A at [5], B at [2] → ordered requires A before B, but pos_A > pos_B → fail
    std::vector<std::vector<uint16_t>> positions = {{5}, {2}};
    REQUIRE(check_phrase_constraint(positions, 5, true) == false);
}

TEST_CASE("PhraseFilter Ordered Multiple Positions", "[ut][ProximityScorer][PhraseFilter]") {
    // terms A at [5, 10], B at [2, 12]
    // Ordered: need pos_A < pos_B. Pair (10, 12) works: span=2, slop=5 → pass
    std::vector<std::vector<uint16_t>> positions = {{5, 10}, {2, 12}};
    REQUIRE(check_phrase_constraint(positions, 5, true) == true);
}

TEST_CASE("PhraseFilter Three Terms Ordered", "[ut][ProximityScorer][PhraseFilter]") {
    // A at [2], B at [5], C at [7]
    // Ordered: 2 < 5 < 7, span = 7-2 = 5, max_span = slop+3-1 = 3+3-1=5 → pass
    std::vector<std::vector<uint16_t>> positions = {{2}, {5}, {7}};
    REQUIRE(check_phrase_constraint(positions, 3, true) == true);
}

TEST_CASE("PhraseFilter Three Terms Ordered Fail", "[ut][ProximityScorer][PhraseFilter]") {
    // A at [2], B at [5], C at [3] → no ordered combination with span ≤ slop+2
    // Only valid ordered combo: A=2, B=5, C=? → C must be > 5, but C only at 3 → fail
    std::vector<std::vector<uint16_t>> positions = {{2}, {5}, {3}};
    REQUIRE(check_phrase_constraint(positions, 3, true) == false);
}

// ===== check_phrase_constraint_sloppy (Lucene normalized slop) tests =====

TEST_CASE("SloppyPhrase Forward Distance Two", "[ut][ProximityScorer][PhraseFilter]") {
    // query offsets [0,1,2], doc pos [0,2,4] → norms [0,1,2] → distance=2.
    // (Aligns with lucene_sloppy_freq_algo.md example 1.)
    std::vector<std::vector<uint16_t>> positions = {{0}, {2}, {4}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 2) == true);
    REQUIRE(check_phrase_constraint_sloppy(positions, 1) == false);
}

TEST_CASE("SloppyPhrase Reverse Costs More Slop", "[ut][ProximityScorer][PhraseFilter]") {
    // query offsets [0,1,2], doc pos [4,2,0] → norms [4,1,-2] → distance=6.
    // (Aligns with lucene_sloppy_freq_algo.md example 2.)
    std::vector<std::vector<uint16_t>> positions = {{4}, {2}, {0}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 6) == true);
    REQUIRE(check_phrase_constraint_sloppy(positions, 5) == false);
}

TEST_CASE("SloppyPhrase Exact Adjacent", "[ut][ProximityScorer][PhraseFilter]") {
    // doc pos [0,1] → norms [0,0] → distance=0 → slop=0 passes.
    std::vector<std::vector<uint16_t>> positions = {{0}, {1}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 0) == true);
}

TEST_CASE("SloppyPhrase Missing Term", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{0}, {}, {2}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 100) == false);
}

TEST_CASE("SloppyPhrase Single Term Passes", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{5}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 0) == true);
}

TEST_CASE("SloppyPhrase Multiple Positions Best Window", "[ut][ProximityScorer][PhraseFilter]") {
    // A at [0, 10], B at [3, 11]. query offsets [0,1].
    // norms: A → {0, 10}, B → {2, 10}. Window {10,10} → distance=0 → slop=0 passes.
    std::vector<std::vector<uint16_t>> positions = {{0, 10}, {3, 11}};
    REQUIRE(check_phrase_constraint_sloppy(positions, 0) == true);
}

// ===== extract_positions_from_token_sequence tests =====

TEST_CASE("ExtractPositions Basic", "[ut][ProximityScorer]") {
    // Document: "苹果(10) 公司(20) 发布(30) 新款(40) 苹果(10) 手机(50) iphone(60)"
    uint32_t token_seq[] = {10, 20, 30, 40, 10, 50, 60};
    uint32_t ids[] = {10, 20, 60};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 7, ids, 3, 64, out);

    REQUIRE(out.size() == 3);
    // term 10 (苹果): positions [0, 4]
    REQUIRE(out[0].size() == 2);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 4);
    // term 20 (公司): positions [1]
    REQUIRE(out[1].size() == 1);
    REQUIRE(out[1][0] == 1);
    // term 60 (iphone): positions [6]
    REQUIRE(out[2].size() == 1);
    REQUIRE(out[2][0] == 6);
}

TEST_CASE("ExtractPositions Ignores Non-ID Tokens", "[ut][ProximityScorer]") {
    // token_seq has terms 30, 40, 50 that are NOT in ids
    uint32_t token_seq[] = {30, 40, 50, 10, 30};
    uint32_t ids[] = {10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 5, ids, 1, 64, out);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].size() == 1);
    REQUIRE(out[0][0] == 3);  // term 10 at pos 3
}

TEST_CASE("ExtractPositions Cap Truncation", "[ut][ProximityScorer]") {
    // term 10 appears 10 times, but cap=4
    uint32_t token_seq[] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    uint32_t ids[] = {10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 10, ids, 1, 4, out);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].size() == 4);  // capped at 4
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 1);
    REQUIRE(out[0][2] == 2);
    REQUIRE(out[0][3] == 3);
}

TEST_CASE("ExtractPositions Null Token Sequence", "[ut][ProximityScorer]") {
    uint32_t ids[] = {10, 20};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(nullptr, 0, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    REQUIRE(out[0].empty());
    REQUIRE(out[1].empty());
}

TEST_CASE("ExtractPositions Empty IDs", "[ut][ProximityScorer]") {
    uint32_t token_seq[] = {10, 20, 30};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 3, nullptr, 0, 64, out);

    REQUIRE(out.empty());
}

TEST_CASE("ExtractPositions Duplicate Term in IDs", "[ut][ProximityScorer]") {
    // Same term_id appears twice in ids (unusual but should handle)
    uint32_t token_seq[] = {10, 20, 10, 30};
    uint32_t ids[] = {10, 10};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 4, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    // Both entries get the same positions
    REQUIRE(out[0].size() == 2);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[0][1] == 2);
    REQUIRE(out[1].size() == 2);
    REQUIRE(out[1][0] == 0);
    REQUIRE(out[1][1] == 2);
}

TEST_CASE("ExtractPositions Term Not In Sequence", "[ut][ProximityScorer]") {
    // ids contains a term that doesn't appear in token_seq at all
    uint32_t token_seq[] = {10, 20, 30};
    uint32_t ids[] = {10, 99};
    std::vector<std::vector<uint16_t>> out;

    extract_positions_from_token_sequence(token_seq, 3, ids, 2, 64, out);

    REQUIRE(out.size() == 2);
    REQUIRE(out[0].size() == 1);
    REQUIRE(out[0][0] == 0);
    REQUIRE(out[1].empty());  // term 99 not found
}

// ===== compute_pairwise_proximity_fast (opt 4.1: ordered O(P^2)->O(P)) tests =====
// Core intent: fast must be bit-for-bit identical to the old function so that
// the ordered path is a pure speedup with zero recall drift. Every case below
// cross-checks fast vs old on the SAME input.

TEST_CASE("FastProximity CrossCheck Ordered Forward", "[ut][ProximityScorer]") {
    std::vector<std::vector<uint16_t>> positions = {{0}, {5}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f / 6.0f);
}

TEST_CASE("FastProximity CrossCheck Ordered Reverse Penalty", "[ut][ProximityScorer]") {
    std::vector<std::vector<uint16_t>> positions = {{10}, {5}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f / 11.0f);
}

TEST_CASE("FastProximity CrossCheck Ordered Multiple Positions", "[ut][ProximityScorer]") {
    // Best forward pair (10,12) → dist=2 → 1/3, same as old ordered branch.
    std::vector<std::vector<uint16_t>> positions = {{10, 100}, {12, 95}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f / 3.0f);
}

TEST_CASE("FastProximity CrossCheck Ordered Reverse Only", "[ut][ProximityScorer]") {
    // No forward pair exists (all A positions > all B positions).
    // Reverse best: (20,18) → gap=2 → dist=4 → 1/5.
    std::vector<std::vector<uint16_t>> positions = {{20, 30}, {5, 18}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f / 5.0f);
}

TEST_CASE("FastProximity CrossCheck Ordered Dense Positions", "[ut][ProximityScorer]") {
    // Dense multi-position lists across three terms — the scenario where the
    // old O(P^2) per-pair loop is slowest. fast must match old exactly.
    std::vector<std::vector<uint16_t>> positions = {
        {1, 7, 15, 40, 88}, {3, 9, 22, 41, 90}, {2, 8, 16, 39, 100}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
}

TEST_CASE("FastProximity CrossCheck Ordered Zero Forward", "[ut][ProximityScorer]") {
    // Equal positions → forward dist=0 → boost 1.0, early-return path.
    std::vector<std::vector<uint16_t>> positions = {{5, 20}, {5, 25}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), true);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f);
}

TEST_CASE("FastProximity CrossCheck Unordered Equals Old", "[ut][ProximityScorer]") {
    // Unordered path reuses the old merge, so results are trivially identical.
    std::vector<std::vector<uint16_t>> positions = {{0, 50, 200}, {3, 48, 300}};
    float old_boost = compute_pairwise_proximity(to_spans(positions), false);
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), false);
    REQUIRE_APPROX(fast_boost, old_boost);
    REQUIRE_APPROX(fast_boost, 1.0f / 3.0f);
}

TEST_CASE("FastProximity CrossCheck Empty Skipped", "[ut][ProximityScorer]") {
    std::vector<std::vector<uint16_t>> positions = {{5, 10}, {}};
    float fast_boost = compute_pairwise_proximity_fast(to_spans(positions), true);
    REQUIRE(fast_boost == 0.0f);
}

// ===== check_phrase_constraint_fast (opt 4.2: ordered O(P^n)->polynomial) tests =====
// Same verdict as check_phrase_constraint; ordered path swaps DFS for greedy.

TEST_CASE("FastPhrase CrossCheck Ordered Pass", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{0}, {2}};
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) ==
            check_phrase_constraint(positions, 5, true));
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) == true);
}

TEST_CASE("FastPhrase CrossCheck Ordered Fail Reverse", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{5}, {2}};
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) ==
            check_phrase_constraint(positions, 5, true));
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) == false);
}

TEST_CASE("FastPhrase CrossCheck Ordered Multiple Positions", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{5, 10}, {2, 12}};
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) ==
            check_phrase_constraint(positions, 5, true));
    REQUIRE(check_phrase_constraint_fast(positions, 5, true) == true);
}

TEST_CASE("FastPhrase CrossCheck Three Terms Ordered Pass", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{2}, {5}, {7}};
    REQUIRE(check_phrase_constraint_fast(positions, 3, true) ==
            check_phrase_constraint(positions, 3, true));
    REQUIRE(check_phrase_constraint_fast(positions, 3, true) == true);
}

TEST_CASE("FastPhrase CrossCheck Three Terms Ordered Fail", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{2}, {5}, {3}};
    REQUIRE(check_phrase_constraint_fast(positions, 3, true) ==
            check_phrase_constraint(positions, 3, true));
    REQUIRE(check_phrase_constraint_fast(positions, 3, true) == false);
}

TEST_CASE("FastPhrase CrossCheck Large Slop Dense Positions",
          "[ut][ProximityScorer][PhraseFilter]") {
    // Dense positions + large slop: the case where the old DFS backtracking
    // balloons. The greedy fast path must reach the identical verdict.
    std::vector<std::vector<uint16_t>> positions = {
        {1, 4, 9, 20, 33, 55}, {2, 6, 10, 25, 40, 60}, {3, 8, 12, 30, 44, 70}};
    for (uint32_t slop : {0u, 2u, 5u, 10u, 50u, 100u}) {
        REQUIRE(check_phrase_constraint_fast(positions, slop, true) ==
                check_phrase_constraint(positions, slop, true));
    }
}

TEST_CASE("FastPhrase CrossCheck Missing Term Fails", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{0}, {}, {2}};
    REQUIRE(check_phrase_constraint_fast(positions, 100, true) == false);
    REQUIRE(check_phrase_constraint_fast(positions, 100, false) == false);
}

TEST_CASE("FastPhrase CrossCheck Single Term Passes", "[ut][ProximityScorer][PhraseFilter]") {
    std::vector<std::vector<uint16_t>> positions = {{5}};
    REQUIRE(check_phrase_constraint_fast(positions, 0, true) == true);
}

TEST_CASE("FastPhrase CrossCheck Unordered Equals Old", "[ut][ProximityScorer][PhraseFilter]") {
    // Unordered path reuses the old sliding window.
    std::vector<std::vector<uint16_t>> positions = {{0, 50}, {3, 48}};
    REQUIRE(check_phrase_constraint_fast(positions, 2, false) ==
            check_phrase_constraint(positions, 2, false));
    REQUIRE(check_phrase_constraint_fast(positions, 2, false) == true);
}
