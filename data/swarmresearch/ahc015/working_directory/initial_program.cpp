# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>

const int GRID_SIZE = 10;
const int NUM_TURNS = 100;
const int NUM_FLAVORS = 3;

const int DR[] = {-1, 1, 0, 0};
const int DC[] = {0, 0, -1, 1};
const char DIR_CHARS[] = {'F', 'B', 'L', 'R'};
const int NUM_DIRECTIONS = 4;

std::array<int, NUM_TURNS> G_FLAVOR_SEQUENCE;
std::array<int, NUM_FLAVORS + 1> G_flavor_total_counts;

struct XorshiftRNG {
    uint64_t x;
    XorshiftRNG() : x(std::chrono::steady_clock::now().time_since_epoch().count()) {}

    uint64_t next() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    }

    int uniform_int(int min_val, int max_val) {
        if (min_val > max_val) return min_val;
        if (min_val == max_val) return min_val;
        uint64_t range = static_cast<uint64_t>(max_val) - min_val + 1;
        return min_val + static_cast<int>(next() % range);
    }
};
XorshiftRNG rng;

auto G_start_time = std::chrono::steady_clock::now();

double elapsed_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - G_start_time).count();
}

struct GameState {
    std::array<std::array<int, GRID_SIZE>, GRID_SIZE> board;
    int num_candies;
    int turn_num_1_indexed;

    GameState() : num_candies(0), turn_num_1_indexed(0) {
        for (int i = 0; i < GRID_SIZE; ++i) {
            board[i].fill(0);
        }
    }

    void place_candy(int r, int c, int flavor) {
        board[r][c] = flavor;
        num_candies++;
    }

    std::pair<int, int> find_pth_empty_cell(int p_1_indexed) const {
        int count = 0;
        for (int r_idx = 0; r_idx < GRID_SIZE; ++r_idx) {
            for (int c_idx = 0; c_idx < GRID_SIZE; ++c_idx) {
                if (board[r_idx][c_idx] == 0) {
                    count++;
                    if (count == p_1_indexed) {
                        return {r_idx, c_idx};
                    }
                }
            }
        }
        return {-1, -1};
    }

    int count_empty_cells() const {
        return GRID_SIZE * GRID_SIZE - num_candies;
    }

    void apply_tilt(int dir_idx) {
        if (dir_idx == 0) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int write_r = 0;
                for (int r = 0; r < GRID_SIZE; ++r) {
                    if (board[r][c] != 0) {
                        if (r != write_r) {
                            board[write_r][c] = board[r][c];
                            board[r][c] = 0;
                        }
                        write_r++;
                    }
                }
            }
        } else if (dir_idx == 1) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int write_r = GRID_SIZE - 1;
                for (int r = GRID_SIZE - 1; r >= 0; --r) {
                    if (board[r][c] != 0) {
                        if (r != write_r) {
                            board[write_r][c] = board[r][c];
                            board[r][c] = 0;
                        }
                        write_r--;
                    }
                }
            }
        } else if (dir_idx == 2) {
            for (int r = 0; r < GRID_SIZE; ++r) {
                int write_c = 0;
                for (int c = 0; c < GRID_SIZE; ++c) {
                    if (board[r][c] != 0) {
                        if (c != write_c) {
                            board[r][write_c] = board[r][c];
                            board[r][c] = 0;
                        }
                        write_c++;
                    }
                }
            }
        } else {
            for (int r = 0; r < GRID_SIZE; ++r) {
                int write_c = GRID_SIZE - 1;
                for (int c = GRID_SIZE - 1; c >= 0; --c) {
                    if (board[r][c] != 0) {
                        if (c != write_c) {
                            board[r][write_c] = board[r][c];
                            board[r][c] = 0;
                        }
                        write_c--;
                    }
                }
            }
        }
    }

    double evaluate_fast() const {
        if (num_candies == 0) return 0.0;

        int same_adj = 0;
        int diff_adj = 0;
        std::array<double, NUM_FLAVORS + 1> sum_r{}, sum_c{};
        std::array<int, NUM_FLAVORS + 1> counts{};

        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                if (board[r][c] == 0) continue;
                int f = board[r][c];
                counts[f]++;
                sum_r[f] += r;
                sum_c[f] += c;
                if (c + 1 < GRID_SIZE) {
                    if (board[r][c + 1] == f) same_adj++;
                    else if (board[r][c + 1] != 0) diff_adj++;
                }
                if (r + 1 < GRID_SIZE) {
                    if (board[r + 1][c] == f) same_adj++;
                    else if (board[r + 1][c] != 0) diff_adj++;
                }
            }
        }

        double compactness = 0.0;
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                if (board[r][c] != 0) {
                    int f = board[r][c];
                    if (counts[f] > 1) {
                        double com_r = sum_r[f] / counts[f];
                        double com_c = sum_c[f] / counts[f];
                        compactness += std::abs(r - com_r) + std::abs(c - com_c);
                    }
                }
            }
        }

        double t = static_cast<double>(turn_num_1_indexed);
        return (80.0 + 3.0 * t) * same_adj
             - (10.0 + 0.5 * t) * diff_adj
             - std::max(0.0, 180.0 - 1.8 * t) * compactness;
    }

    double evaluate() const {
        if (num_candies == 0) return 0.0;

        long long total_sq_sum = 0;
        std::array<std::array<bool, GRID_SIZE>, GRID_SIZE> visited;
        for (int i = 0; i < GRID_SIZE; ++i) visited[i].fill(false);
        std::array<std::pair<int, int>, GRID_SIZE * GRID_SIZE> q_arr;

        for (int r_start = 0; r_start < GRID_SIZE; ++r_start) {
            for (int c_start = 0; c_start < GRID_SIZE; ++c_start) {
                if (board[r_start][c_start] != 0 && !visited[r_start][c_start]) {
                    int current_flavor = board[r_start][c_start];
                    long long comp_size = 0;
                    q_arr[0] = {r_start, c_start};
                    visited[r_start][c_start] = true;
                    int head = 0, tail = 1;

                    while (head < tail) {
                        comp_size++;
                        int curr_r = q_arr[head].first;
                        int curr_c = q_arr[head].second;
                        head++;
                        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
                            int nr = curr_r + DR[i];
                            int nc = curr_c + DC[i];
                            if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE &&
                                !visited[nr][nc] && board[nr][nc] == current_flavor) {
                                visited[nr][nc] = true;
                                q_arr[tail++] = {nr, nc};
                            }
                        }
                    }
                    total_sq_sum += comp_size * comp_size;
                }
            }
        }

        int same_adj = 0;
        int diff_adj = 0;
        std::array<double, NUM_FLAVORS + 1> sum_r{}, sum_c{};
        std::array<int, NUM_FLAVORS + 1> counts{};

        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                if (board[r][c] == 0) continue;
                int f = board[r][c];
                counts[f]++;
                sum_r[f] += r;
                sum_c[f] += c;
                if (c + 1 < GRID_SIZE) {
                    if (board[r][c + 1] == f) same_adj++;
                    else if (board[r][c + 1] != 0) diff_adj++;
                }
                if (r + 1 < GRID_SIZE) {
                    if (board[r + 1][c] == f) same_adj++;
                    else if (board[r + 1][c] != 0) diff_adj++;
                }
            }
        }

        double compactness = 0.0;
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                if (board[r][c] != 0) {
                    int f = board[r][c];
                    if (counts[f] > 1) {
                        double com_r = sum_r[f] / counts[f];
                        double com_c = sum_c[f] / counts[f];
                        compactness += std::abs(r - com_r) + std::abs(c - com_c);
                    }
                }
            }
        }

        double separation = 0.0;
        std::array<double, NUM_FLAVORS + 1> com_r{}, com_c{};
        for (int f = 1; f <= NUM_FLAVORS; ++f) {
            if (counts[f] > 0) {
                com_r[f] = sum_r[f] / counts[f];
                com_c[f] = sum_c[f] / counts[f];
            }
        }
        for (int f1 = 1; f1 <= NUM_FLAVORS; ++f1) {
            if (counts[f1] == 0) continue;
            for (int f2 = f1 + 1; f2 <= NUM_FLAVORS; ++f2) {
                if (counts[f2] == 0) continue;
                separation += std::abs(com_r[f1] - com_r[f2]) + std::abs(com_c[f1] - com_c[f2]);
            }
        }

        double t = static_cast<double>(turn_num_1_indexed);

        double conn_weight = 15.0 + 1.5 * t;
        double adj_weight = 80.0 + 3.0 * t;
        double diff_adj_weight = -(10.0 + 0.5 * t);
        double compactness_weight = -std::max(0.0, 180.0 - 1.8 * t);
        double separation_weight = std::max(0.0, 40.0 - 0.3 * t);

        return conn_weight * total_sq_sum
             + adj_weight * same_adj
             + diff_adj_weight * diff_adj
             + compactness_weight * compactness
             + separation_weight * separation;
    }
};

struct LookaheadConfig {
    int max_depth;
    int samples[3];
    bool use_fast_at_leaf;
};

double eval_lookahead(const GameState& state_after_tilt, int turn_T_of_candy_just_processed, int depth_remaining, const LookaheadConfig& config);

char decide_tilt_direction_logic(const GameState& current_gs_after_placement) {
    double best_overall_eval = std::numeric_limits<double>::lowest();
    int best_dir_idx = 0;

    int turn_T = current_gs_after_placement.turn_num_1_indexed;

    double elapsed = elapsed_seconds();
    double remaining = 1.92 - elapsed;
    int turns_left = NUM_TURNS - turn_T + 1;
    double time_per_turn = remaining / turns_left;

    LookaheadConfig config;
    if (turn_T <= 12 && time_per_turn > 0.016) {
        config.max_depth = 3;
        config.samples[0] = 10;
        config.samples[1] = 5;
        config.samples[2] = 3;
        config.use_fast_at_leaf = true;
    } else if (time_per_turn > 0.008) {
        config.max_depth = 2;
        config.samples[0] = 24;
        config.samples[1] = 6;
        config.samples[2] = 0;
        config.use_fast_at_leaf = false;
    } else {
        config.max_depth = 2;
        config.samples[0] = 14;
        config.samples[1] = 4;
        config.samples[2] = 0;
        config.use_fast_at_leaf = false;
    }

    for (int i = 0; i < NUM_DIRECTIONS; ++i) {
        GameState gs_after_tilt = current_gs_after_placement;
        gs_after_tilt.apply_tilt(i);

        double eval = eval_lookahead(gs_after_tilt, turn_T, config.max_depth, config);

        if (eval > best_overall_eval) {
            best_overall_eval = eval;
            best_dir_idx = i;
        }
    }
    return DIR_CHARS[best_dir_idx];
}


double eval_lookahead(const GameState& state_after_tilt, int turn_T_of_candy_just_processed, int depth_remaining, const LookaheadConfig& config) {
    if (depth_remaining == 0 || turn_T_of_candy_just_processed == NUM_TURNS) {
        if (config.use_fast_at_leaf) return state_after_tilt.evaluate_fast();
        return state_after_tilt.evaluate();
    }

    int num_empty = state_after_tilt.count_empty_cells();
    if (num_empty == 0) {
        if (config.use_fast_at_leaf) return state_after_tilt.evaluate_fast();
        return state_after_tilt.evaluate();
    }

    int next_candy_flavor = G_FLAVOR_SEQUENCE[turn_T_of_candy_just_processed];
    int sample_count_param_idx = config.max_depth - depth_remaining;
    int sample_count_this_depth = config.samples[sample_count_param_idx];
    int actual_num_samples = std::min(sample_count_this_depth, num_empty);

    if (actual_num_samples == 0) {
        if (config.use_fast_at_leaf) return state_after_tilt.evaluate_fast();
        return state_after_tilt.evaluate();
    }

    double sum_over_sampled_placements = 0.0;
    for (int s = 0; s < actual_num_samples; ++s) {
        int p_val;
        if (actual_num_samples == num_empty) {
            p_val = s + 1;
        } else {
            p_val = rng.uniform_int(1, num_empty);
        }

        GameState S_after_placement = state_after_tilt;
        std::pair<int, int> candy_loc = S_after_placement.find_pth_empty_cell(p_val);
        S_after_placement.place_candy(candy_loc.first, candy_loc.second, next_candy_flavor);
        S_after_placement.turn_num_1_indexed = turn_T_of_candy_just_processed + 1;

        double max_eval = std::numeric_limits<double>::lowest();
        for (int dir_idx = 0; dir_idx < NUM_DIRECTIONS; ++dir_idx) {
            GameState S_after_tilt = S_after_placement;
            S_after_tilt.apply_tilt(dir_idx);
            double val = eval_lookahead(S_after_tilt, S_after_placement.turn_num_1_indexed, depth_remaining - 1, config);
            if (val > max_eval) {
                max_eval = val;
            }
        }
        sum_over_sampled_placements += max_eval;
    }

    return sum_over_sampled_placements / actual_num_samples;
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    G_flavor_total_counts.fill(0);
    for (int t = 0; t < NUM_TURNS; ++t) {
        std::cin >> G_FLAVOR_SEQUENCE[t];
        G_flavor_total_counts[G_FLAVOR_SEQUENCE[t]]++;
    }
    G_start_time = std::chrono::steady_clock::now();

    GameState current_gs;
    for (int t = 0; t < NUM_TURNS; ++t) {
        current_gs.turn_num_1_indexed = t + 1;

        int p_val;
        std::cin >> p_val;

        std::pair<int, int> candy_loc = current_gs.find_pth_empty_cell(p_val);
        current_gs.place_candy(candy_loc.first, candy_loc.second, G_FLAVOR_SEQUENCE[t]);

        char chosen_dir = decide_tilt_direction_logic(current_gs);

        std::cout << chosen_dir << std::endl;

        int dir_idx = 0;
        for (int k = 0; k < NUM_DIRECTIONS; ++k) {
            if (DIR_CHARS[k] == chosen_dir) {
                dir_idx = k;
                break;
            }
        }
        current_gs.apply_tilt(dir_idx);
    }

    return 0;
}
# EVOLVE-BLOCK-END
