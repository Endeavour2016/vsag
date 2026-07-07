
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

#pragma once

#include <cstdint>
#include <vector>

namespace vsag {

// Non-owning view over a contiguous, immutable position list. Used by the
// proximity hot path to avoid per-candidate vector copies; the underlying data
// lives in SparseTermDataCell::term_pos_pool_ and is read-only during a query.
struct PosSpan {
    const uint16_t* data{nullptr};
    uint32_t size{0};

    bool
    empty() const {
        return size == 0;
    }
    const uint16_t*
    begin() const {
        return data;
    }
    const uint16_t*
    end() const {
        return data + size;
    }
    uint16_t
    operator[](uint32_t i) const {
        return data[i];
    }
};

// Compute pairwise proximity boost for a set of query terms' position lists.
//
// For each pair of terms (i, j) where i < j, finds the minimum distance between
// their position lists and accumulates 1/(min_dist + 1).
//
// When ordered=true, the distance for reversed pairs (pos_i > pos_j) is doubled
// as a penalty (similar to Lucene's slop cost for reversals).
//
// Returns 0.0 if fewer than 2 non-empty position lists are provided.
float
compute_pairwise_proximity(const std::vector<PosSpan>& position_lists, bool ordered);

// Faster equivalent of compute_pairwise_proximity that returns the same boost.
//
// Scores the same C(n,2) term pairs and accumulates the same 1/(min_dist + 1),
// so the result is bit-for-bit identical to compute_pairwise_proximity. The only
// difference is the ordered per-pair distance: instead of the O(P^2) all-pairs
// double loop it uses a two-pass two-pointer merge over the sorted lists, cutting
// a single ordered pair from O(P^2) to O(P). The unordered path is unchanged (it
// was already an O(L) merge).
float
compute_pairwise_proximity_fast(const std::vector<PosSpan>& position_lists, bool ordered);

// Simplified proximity boost that only scores adjacent query-term pairs.
//
// Unlike compute_pairwise_proximity (which scores all C(n,2) pairs), this scores
// only the n-1 pairs (i, i+1), where adjacency follows the order of
// position_lists (i.e. the query's raw_query_.ids_ order). For query ABC it
// scores (A,B) and (B,C) but not (A,C).
//
// The ordered flag and per-pair distance/boost semantics match
// compute_pairwise_proximity. Empty position lists contribute nothing.
float
calculate_pairwise_proximity(const std::vector<PosSpan>& position_lists, bool ordered);

// Check if a set of terms satisfy a phrase constraint.
//
// All terms in phrase_term_positions must be present (non-empty position list).
// The minimum span covering one position from each term must be <= slop + num_terms - 1.
// If ordered=true, the chosen positions must also be in strictly increasing order.
//
// Returns true if the constraint is satisfied, false otherwise.
bool
check_phrase_constraint(const std::vector<std::vector<uint16_t>>& phrase_term_positions,
                        uint32_t slop,
                        bool ordered);

// Faster equivalent of check_phrase_constraint that returns the same verdict.
//
// Same semantics as check_phrase_constraint (all terms must be present; a match
// needs a window with span <= slop + num_terms - 1; ordered=true additionally
// requires strictly increasing positions). The unordered path reuses the same
// O(M log M) sliding window. The ordered path replaces the worst-case O(P^n) DFS
// backtracking with a greedy + binary-search scan: for each start position of
// term 0, chain the smallest strictly-larger position of each subsequent term via
// std::lower_bound, giving the minimal reachable end; pass if any start's span
// fits. This is O(L0 * n * log L) and yields the identical true/false result.
bool
check_phrase_constraint_fast(const std::vector<std::vector<uint16_t>>& phrase_term_positions,
                             uint32_t slop,
                             bool ordered);

// Phrase constraint using Lucene SloppyPhraseMatcher's normalized-window slop.
//
// Each term's positions are normalized by their query offset (the term's index
// in phrase_term_positions): norm = doc_pos - term_idx. The match distance is
// max(norm) - min(norm) over a window covering one position per term; the
// constraint passes if any such window has distance <= slop. This encodes order
// into the offsets (reversals cost extra slop) so there is no ordered/unordered
// distinction. Returns true as soon as one satisfying window is found.
//
// All terms must be present (non-empty). Returns true for 0 or 1 terms.
bool
check_phrase_constraint_sloppy(const std::vector<std::vector<uint16_t>>& phrase_term_positions,
                               uint32_t slop);

// Extract per-term positions from a raw token sequence.
//
// For each term in ids[0..ids_len), scans token_sequence to find all positions
// where that term appears. Caps each term's positions at max_positions_per_term.
// Only terms present in ids[] are extracted; other tokens are ignored.
//
// out_positions is resized to ids_len, with out_positions[i] containing the
// positions for ids[i].
void
extract_positions_from_token_sequence(const uint32_t* token_sequence,
                                      uint32_t seq_len,
                                      const uint32_t* ids,
                                      uint32_t ids_len,
                                      uint32_t max_positions_per_term,
                                      std::vector<std::vector<uint16_t>>& out_positions);

}  // namespace vsag
