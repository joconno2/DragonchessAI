#pragma once

#include "game.h"
#include <vector>

namespace dragonchess {

// Number of features in the TD feature vector.
// All features are computed from Gold's perspective (positive = good for Gold).
constexpr int NUM_TD_FEATURES = 40;

// Extract a feature vector from the current game state.
// Feature layout (Gold-positive perspective):
//   [0-13]  Material count diff per piece type (gold_count - scarlet_count)
//           Types: Sylph(1), Griffin(2), Dragon(3), Oliphant(4), Unicorn(5),
//                  Hero(6), Thief(7), Cleric(8), Mage(9), Paladin(11),
//                  Warrior(12), Basilisk(13), Elemental(14), Dwarf(15)
//   [14-16] Level piece count diff (gold - scarlet) per level: Sky, Ground, Cavern
//   [17-19] Level material value diff (gold - scarlet) per level: Sky, Ground, Cavern
//   [20]    Gold mean piece row on Ground (higher = more advanced toward row 7)
//   [21]    Scarlet mean piece advancement on Ground (7 - mean_row)
//   [22]    Gold mean piece row in Cavern
//   [23]    Scarlet mean piece advancement in Cavern (7 - mean_row)
//   [24]    Scarlet pieces within Chebyshev distance 2 of gold king (king danger)
//   [25]    Gold pieces within Chebyshev distance 2 of scarlet king (attack proximity)
//   [26]    Gold frozen piece count
//   [27]    Scarlet frozen piece count
//   [28]    Cross-level piece count diff (gold - scarlet)
//           Cross-level: Dragon(3), Griffin(2), Paladin(11), Hero(6), Mage(9)
//   [29]    Ground center control diff: gold - scarlet pieces in cols 3-8
//   [30]    Total piece count diff (gold - scarlet), normalized by 26
//   [31]    Gold king row on Ground (0 if king not on Ground level)
//   [32]    Scarlet king advancement on Ground: 7-row (0 if king not on Ground)
//   [33]    Gold absolute material value on Sky level
//   [34]    Scarlet absolute material value on Sky level
//   [35]    Gold absolute material value in Cavern level
//   [36]    Scarlet absolute material value in Cavern level
//   [37]    Game progress: min(move_count / 200, 1.0)
//   [38]    Gold has Dragon: 1.0 if present, 0.0 otherwise
//   [39]    Scarlet has Dragon: 1.0 if present, 0.0 otherwise
std::vector<float> extract_td_features(const Game& game);

} // namespace dragonchess
