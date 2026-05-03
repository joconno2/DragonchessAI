#include "mcts.h"
#include <numeric>
#include <iostream>
#include <sstream>

namespace dragonchess {

// ---------------------------------------------------------------------------
// MCTS implementation
// ---------------------------------------------------------------------------

MCTSNode* MCTS::search(Game& game, int num_simulations) {
    root_ = std::make_unique<MCTSNode>();

    // Initial expansion of root
    auto eval = nn_.evaluate(game);
    expand(root_.get(), game, eval.move_priors);
    add_dirichlet_noise(root_.get());

    // Build action -> Move map for current position
    action_to_move_map_.clear();
    auto legal = game.get_all_moves();
    for (const auto& m : legal)
        action_to_move_map_[move_to_action(m)] = m;

    for (int i = 0; i < num_simulations; ++i) {
        Game sim = game;  // copy for simulation
        simulate(root_.get(), sim, true);
    }

    return root_.get();
}

float MCTS::simulate(MCTSNode* node, Game& game, bool is_root) {
    // Terminal check
    if (game.game_over || game.get_all_moves().empty()) {
        // Determine outcome from Gold's perspective
        if (game.game_over) {
            // The player who just moved won (opponent of current_turn)
            return (game.current_turn == Color::GOLD) ? -1.0f : 1.0f;
        }
        return 0.0f;  // stalemate
    }

    // Check no-progress draw
    if (game.no_capture_count >= 100 || static_cast<int>(game.game_log.size()) >= 1000) {
        return 0.0f;
    }

    if (!node->expanded) {
        // Leaf: expand and evaluate
        auto eval = nn_.evaluate(game);
        expand(node, game, eval.move_priors);
        // Value from current player's perspective -> convert to Gold's perspective
        float v = eval.value;
        if (game.current_turn == Color::SCARLET) v = -v;
        node->visit_count++;
        node->value_sum += v;
        return v;
    }

    // Select child via PUCT
    MCTSNode* child = select_child(node);
    if (!child) return 0.0f;

    // Make the child's move
    auto it = action_to_move_map_.find(child->action);
    if (it == action_to_move_map_.end()) {
        // Rebuild action map for this position
        auto legal = game.get_all_moves();
        action_to_move_map_.clear();
        for (const auto& m : legal)
            action_to_move_map_[move_to_action(m)] = m;
        it = action_to_move_map_.find(child->action);
        if (it == action_to_move_map_.end()) return 0.0f;
    }

    game.make_move(it->second);
    game.update();

    float value = simulate(child, game, false);

    game.undo_move();

    // Backpropagate (value is from Gold's perspective)
    node->visit_count++;
    node->value_sum += value;

    return value;
}

MCTSNode* MCTS::select_child(MCTSNode* node) const {
    float best_score = -1e30f;
    MCTSNode* best = nullptr;
    float sqrt_parent = std::sqrt(static_cast<float>(node->visit_count));

    for (auto& child : node->children) {
        float q = child->q_value();
        // Flip Q for the opponent's perspective
        // Values are stored from Gold's perspective.
        // If it's Scarlet's turn at this node, Scarlet wants to minimize Gold's value.
        // PUCT: the selecting player wants to maximize their own outcome.
        // We'll handle perspective in the value storage instead.

        float u = c_puct_ * child->prior * sqrt_parent / (1.0f + child->visit_count);
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }
    return best;
}

void MCTS::expand(MCTSNode* node, const Game& game,
                  const std::vector<std::pair<int, float>>& priors) {
    node->expanded = true;
    node->children.reserve(priors.size());
    for (const auto& [action, prob] : priors) {
        auto child = std::make_unique<MCTSNode>();
        child->action = action;
        child->prior = prob;
        node->children.push_back(std::move(child));
    }
}

void MCTS::add_dirichlet_noise(MCTSNode* node) {
    if (node->children.empty()) return;

    int n = static_cast<int>(node->children.size());
    std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
    std::vector<float> noise(n);
    float noise_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        noise[i] = gamma(rng_);
        noise_sum += noise[i];
    }
    if (noise_sum > 0) {
        for (int i = 0; i < n; ++i)
            noise[i] /= noise_sum;
    }

    for (int i = 0; i < n; ++i) {
        node->children[i]->prior =
            (1.0f - dirichlet_frac_) * node->children[i]->prior +
            dirichlet_frac_ * noise[i];
    }
}

std::vector<std::pair<int, float>> MCTS::get_policy(
    const MCTSNode* root, float temperature) const
{
    std::vector<std::pair<int, float>> policy;
    policy.reserve(root->children.size());

    if (temperature < 0.01f) {
        // Argmax
        int best_visits = -1;
        int best_action = -1;
        for (const auto& child : root->children) {
            if (child->visit_count > best_visits) {
                best_visits = child->visit_count;
                best_action = child->action;
            }
        }
        for (const auto& child : root->children) {
            policy.push_back({child->action,
                             child->action == best_action ? 1.0f : 0.0f});
        }
    } else {
        // Temperature-scaled visit counts
        float inv_temp = 1.0f / temperature;
        float max_log_count = -1e30f;
        for (const auto& child : root->children) {
            if (child->visit_count > 0) {
                float lc = inv_temp * std::log(static_cast<float>(child->visit_count));
                max_log_count = std::max(max_log_count, lc);
            }
        }

        float sum = 0.0f;
        for (const auto& child : root->children) {
            float p = 0.0f;
            if (child->visit_count > 0) {
                p = std::exp(inv_temp * std::log(static_cast<float>(child->visit_count))
                             - max_log_count);
            }
            policy.push_back({child->action, p});
            sum += p;
        }
        if (sum > 0) {
            for (auto& [a, p] : policy)
                p /= sum;
        }
    }

    return policy;
}

int MCTS::select_action(const std::vector<std::pair<int, float>>& policy) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    for (const auto& [action, prob] : policy) {
        cumsum += prob;
        if (r <= cumsum) return action;
    }
    return policy.back().first;
}

// ---------------------------------------------------------------------------
// Self-play
// ---------------------------------------------------------------------------

SelfPlayGame run_mcts_selfplay(const DualHeadWeights& nn,
                                int num_simulations,
                                float temperature,
                                int temp_drop_move,
                                int max_moves) {
    SelfPlayGame result;
    Game game;
    MCTS mcts(nn);

    for (int move_num = 0; move_num < max_moves; ++move_num) {
        if (game.game_over || game.get_all_moves().empty())
            break;
        if (game.no_capture_count >= 100 || static_cast<int>(game.game_log.size()) >= 1000)
            break;

        // Record position features
        SelfPlayPosition pos;
        pos.features = extract_td_features_sparse(game);
        pos.turn = game.current_turn;

        // Run MCTS
        float temp = (move_num < temp_drop_move) ? temperature : 0.01f;
        MCTSNode* root = mcts.search(game, num_simulations);
        auto policy = mcts.get_policy(root, temp);
        pos.mcts_policy = policy;

        result.positions.push_back(std::move(pos));

        // Select and play move
        int action = mcts.select_action(policy);

        // Find the actual Move for this action
        auto legal = game.get_all_moves();
        bool found = false;
        for (const auto& m : legal) {
            if (move_to_action(m) == action) {
                game.make_move(m);
                game.update();
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    // Determine outcome
    if (game.game_over) {
        // Last player to move won
        result.outcome = (game.current_turn == Color::GOLD) ? -1.0f : 1.0f;
    } else {
        result.outcome = 0.0f;
    }

    result.num_moves = static_cast<int>(result.positions.size());

    // Fill in outcomes for each position (from that player's perspective)
    for (auto& pos : result.positions) {
        if (pos.turn == Color::GOLD)
            pos.outcome = result.outcome;
        else
            pos.outcome = -result.outcome;
    }

    return result;
}

} // namespace dragonchess
