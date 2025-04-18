#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include "thread_pool.h"

using namespace std;

static bool g_use_hash = true;

class Configuration;
void log_squares(Configuration& C);

const double PI = M_PI;
const double INF = numeric_limits<double>::infinity();
const unsigned DEFAULT_WORKERS = 8;

template <typename T> 
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

random_device rd;
mt19937 gen(rd());

// Simple hash functor for a (i,j) pair
struct PairHash {
    size_t operator()(const pair<int,int>& p) const {
        // A simple combination of hashes
        auto h1 = std::hash<int>()(p.first);
        auto h2 = std::hash<int>()(p.second);
        // Combine them in a typical way
        return (h1 * 31ULL) ^ h2;
    }
};

ThreadPool& getGlobalThreadPool()
{
    // static ThreadPool pool(std::max(DEFAULT_WORKERS, thread::hardware_concurrency()));
    static ThreadPool pool(DEFAULT_WORKERS);
    return pool;
}

// Structure to represent a configuration of squares
class Configuration {
public:
    int n;
    double L;
    vector<array<double, 3>> squares; // each [x,y,theta]
    vector<double> cos_table;
    vector<double> sin_table;

    vector<vector<double>> pair_inflations;
    vector<double> boundary_inflations;
    double max_inflation;

    vector<pair<int,int>> inflation_pairs;

    // spatial hash fields
    bool use_spatial_hash;
    vector<double> min_inflations;
    vector<unordered_set<int>> guilt_sets;
    vector<int> accused;

    vector<pair<int, double>> temp_min_inflations;
    vector<pair<int, int>> temp_blames;

    double cell_size;
    int grid_dim;
    vector<vector<vector<int>>> grid;

    // Cleared at the start of propose_replacement
    unordered_map<pair<int,int>, double, PairHash> pair_inflation_cache;

    Configuration(int num_squares, double length, bool use_spatial_hash_method=g_use_hash)
        : n(num_squares),
          L(length),
          pair_inflations(num_squares+1, vector<double>(num_squares+1, 0.0)),
          boundary_inflations(num_squares+1, 0),
          max_inflation(INF),
          use_spatial_hash(use_spatial_hash_method)
    {
        inflation_pairs.reserve(n*(n-1)/2);
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                inflation_pairs.push_back({i, j});
            }
        }

        // Initialize squares to random positions & orientations
        uniform_real_distribution<> pos_dist(-L, L);
        uniform_real_distribution<> angle_dist(0.0, 2 * PI);

        for (int i = 0; i < n; i++)
        {
            double x = pos_dist(gen);
            double y = pos_dist(gen);
            double theta = angle_dist(gen);
            squares.push_back({x, y, theta});
            cos_table.push_back(cos(theta));
            sin_table.push_back(sin(theta));
        }

        // Extra square for scratchpad (used for algorithm)
        squares.push_back({0.0, 0.0, 0.0});
        cos_table.push_back(1.0);
        sin_table.push_back(0.0);

        if (use_spatial_hash)
        {
            min_inflations.resize(n, INF);
            guilt_sets.resize(n);
            accused.resize(n, -1);

            grid_dim = (int) ceil(sqrt((double) n));
            cell_size = (2.0 * L) / (double) grid_dim;

            grid.resize(grid_dim);
            for (int r = 0; r < grid_dim; r++)
                grid[r].resize(grid_dim);
        }

        // Calculate pairwise and boundary inflations
        update_inflations_full();
    }

    Configuration(const Configuration& other) 
      : n(other.n),
        L(other.L),
        squares(other.squares),
        cos_table(other.cos_table),
        sin_table(other.sin_table),
        pair_inflations(other.pair_inflations),
        boundary_inflations(other.boundary_inflations),
        max_inflation(other.max_inflation),
        inflation_pairs(other.inflation_pairs),
        use_spatial_hash(other.use_spatial_hash),
        min_inflations(other.min_inflations),
        guilt_sets(other.guilt_sets),
        accused(other.accused),
        temp_min_inflations(other.temp_min_inflations),
        temp_blames(other.temp_blames),
        cell_size(other.cell_size),
        grid_dim(other.grid_dim),
        grid(other.grid),
        pair_inflation_cache() // copy won't matter, we'll clear on new proposals anyway
    {}

    void set_square(int i, double a, double b, double theta)
    {
        squares[i][0] = a;
        squares[i][1] = b;
        squares[i][2] = theta;
        cos_table[i] = cos(theta);
        sin_table[i] = sin(theta);
    }

    // The main geometry routine for squares at (0,0,0) vs. (a,b,theta)
    inline double psi_0(double a, double b, double theta)
    {
        double numer = std::abs(a) + std::abs(b);
        double ab_sgn = (a*b > 0.0) ? 1.0 : (a*b < 0.0 ? -1.0 : 0.0);

        double min_val = INF;
        for (int i = 0; i < 4; i++)
        {
            double angle = theta + i*(PI/2) + PI/4;
            double sin_val = sin(angle);
            double denom = std::abs(1 - sqrt(2) * ab_sgn * sin_val);
            double val = numer / denom;
            if (val < min_val) {
                min_val = val;
            }
        }
        return min_val;
    }

    // Basic pair-inflation function (used by cache)
    inline double get_pair_inflation_raw(int i, int j)
    {
        // i->j
        double a = (squares[j][0] - squares[i][0]) * cos_table[i]
                 + (squares[j][1] - squares[i][1]) * sin_table[i];
        double b = -(squares[j][0] - squares[i][0]) * sin_table[i]
                 + (squares[j][1] - squares[i][1]) * cos_table[i];
        double t = squares[j][2] - squares[i][2];
        double psi_1 = psi_0(a, b, t);

        // j->i
        double a2 = (squares[i][0] - squares[j][0]) * cos_table[j]
                  + (squares[i][1] - squares[j][1]) * sin_table[j];
        double b2 = -(squares[i][0] - squares[j][0]) * sin_table[j]
                  + (squares[i][1] - squares[j][1]) * cos_table[j];
        double t2 = squares[i][2] - squares[j][2];
        double psi_2 = psi_0(a2, b2, t2);

        return std::max(psi_1, psi_2);
    }

    // Cached version of get_pair_inflation
    inline double get_pair_inflation_cached(int i, int j)
    {
        if (i > j) std::swap(i, j);
        auto key = make_pair(i, j);

        auto it = pair_inflation_cache.find(key);
        if (it != pair_inflation_cache.end()) {
            return it->second;
        }
        double val = get_pair_inflation_raw(i, j);
        pair_inflation_cache[key] = val;
        return val;
    }

    // For places that do not use the cache
    inline double get_pair_inflation(int i, int j)
    {
        return get_pair_inflation_raw(i, j);
    }

    // Calculate the inflation of square i against the boundary
    inline double get_boundary_inflation(int i)
    {
        double numer = L - max(abs(squares[i][0]), abs(squares[i][1]));
        double denom = max(abs(cos_table[i]), abs(sin_table[i]));
        return numer / denom;
    }

    // Recalculate all pairwise inflations (non-hash path)
    void update_all_pair_inflations()
    {
        for (int idx = 0; idx < (int)inflation_pairs.size(); idx++) {
            auto [i, j] = inflation_pairs[idx];
            pair_inflations[i][j] = get_pair_inflation_raw(i, j);
        }

        for (auto &pr : inflation_pairs) {
            int i = pr.first;
            int j = pr.second;
            max_inflation = min(max_inflation, pair_inflations[i][j]);
        }
    }

    // Recalculate boundary inflations
    void update_all_boundary_inflations()
    {
        for (int i = 0; i < n; i++) 
        {
            boundary_inflations[i] = get_boundary_inflation(i);
            max_inflation = std::min(max_inflation, boundary_inflations[i]);
        }
    }

    pair<int, int> cell_indices(double x, double y) const
    {
        int cx = (int) floor((x + L) / cell_size);
        int cy = (int) floor((y + L) / cell_size);
        return {cx, cy};
    }

    void insert_into_spatial_hash(int i)
    {
        auto [cx, cy] = cell_indices(squares[i][0], squares[i][1]);
        grid[cx][cy].push_back(i);
    }

    void remove_from_spatial_hash(int i)
    {
        auto [cx, cy] = cell_indices(squares[i][0], squares[i][1]);
        auto &bucket = grid[cx][cy];
        for (int k = 0; k < (int)bucket.size(); k++) {
            if (bucket[k] == i) {
                bucket[k] = bucket.back();
                bucket.pop_back();
                break;
            }
        }
    }

    pair<double, int> find_min_inflation_spatial(int i, int j, double old_min)
    {
        // boundary inflation for i
        double boundary_infl = boundary_inflations[i];
        double local_min_infl = std::min(old_min, boundary_infl);
        int inflicted_by = -1;

        double ix = squares[i][0];
        double iy = squares[i][1];
        auto [cx, cy] = cell_indices(ix, iy);

        double cs = cell_size;
        // bottom-left corner of the cell
        double a = (double) cx * cs - L; 
        double b = (double) cy * cs - L;

        // distance from i to cell boundary, used in ring break
        double d = min(min(ix - a, a + cs - ix), min(iy - b, b + cs - iy));

        // 0-ring
        for (int sq : grid[cx][cy]) {
            if (sq == i || sq == j) continue;

            double dx = squares[i][0] - squares[sq][0];
            double dy = squares[i][1] - squares[sq][1];
            double dist2 = dx*dx + dy*dy;
            // bounding distance check
            if (dist2 > 4.0 * local_min_infl * local_min_infl) {
                continue;
            }

            double infl = get_pair_inflation_cached(i, sq);
            if (infl < local_min_infl) {
                local_min_infl = infl;
                inflicted_by = sq;
            }
        }

        // rings out to grid_dim, or until we can't possibly find a smaller inflation
        for (int ring = 1; ring <= grid_dim; ring++)
        {
            int cells_to_check = 8 * ring;
            if (2.0 * local_min_infl < d + (ring-1)*cs) {
                break;
            }

            for (int ring_i = 0; ring_i < cells_to_check; ring_i++)
            {
                int side = ring_i / (2 * ring);
                int side_idx = ring_i % (2 * ring);

                int gx = cx;
                int gy = cy;

                // compute the cell indices
                if (side == 0) {
                    gx = cx - ring + side_idx;
                    gy = cy + ring;
                } else if (side == 1) {
                    gx = cx + ring;
                    gy = cy + ring - side_idx;
                } else if (side == 2) {
                    gx = cx + ring - side_idx;
                    gy = cy - ring;
                } else {
                    gx = cx - ring;
                    gy = cy - ring + side_idx;
                }

                if (gx < 0 || gx >= grid_dim || gy < 0 || gy >= grid_dim) 
                    continue;

                for (int sq : grid[gx][gy]) {
                    if (sq == i || sq == j) continue;

                    double dx = squares[i][0] - squares[sq][0];
                    double dy = squares[i][1] - squares[sq][1];
                    double dist2 = dx*dx + dy*dy;
                    if (dist2 > 4.0 * local_min_infl * local_min_infl) {
                        continue;
                    }

                    double infl = get_pair_inflation_cached(i, sq);
                    if (infl < local_min_infl) {
                        local_min_infl = infl;
                        inflicted_by = sq;
                    }
                }
            }
        }

        return make_pair(local_min_infl, inflicted_by);
    }

    void update_inflations_full_spatialhash()
    {
        // Clear the grid
        for (int r = 0; r < grid_dim; r++) {
            for (int c = 0; c < grid_dim; c++) {
                grid[r][c].clear();
            }
        }

        // Insert squares
        for (int i = 0; i < n; i++) {
            insert_into_spatial_hash(i);
            min_inflations[i] = INF;
            guilt_sets[i].clear();
            accused[i] = -1;
        }

        // For each square, compute boundary & min_infl
        for (int i = 0; i < n; i++)
        {
            boundary_inflations[i] = get_boundary_inflation(i);
            auto [m_infl, inflicted_by] = find_min_inflation_spatial(i, n, INF);
            min_inflations[i] = m_infl;
            if (inflicted_by >= 0) {
                accused[i] = inflicted_by;
                guilt_sets[inflicted_by].insert(i);
            }
            if (m_infl < max_inflation) {
                max_inflation = m_infl;
            }
        }
    }

    // Recalculate all inflations
    void update_inflations_full()
    {
        max_inflation = INF;

        if (!use_spatial_hash)
        {
            update_all_pair_inflations();
            update_all_boundary_inflations();
        } 
        else 
        {
            update_inflations_full_spatialhash();
        }
    }

    double propose_replacement(int i, double a, double b, double theta)
    {
        pair_inflation_cache.clear();

        if (use_spatial_hash)
            remove_from_spatial_hash(n);

        double i_inflation = INF;

        // Put the proposed square in the scratch index n
        set_square(n, a, b, theta);

        // boundary inflation
        double bound_inflation = get_boundary_inflation(n);
        boundary_inflations[n] = bound_inflation;
        i_inflation = min(i_inflation, bound_inflation);

        if (!use_spatial_hash)
        {
            for (int idx = 0; idx < n-1; idx++) {
                int j = (idx >= i) ? idx+1 : idx;
                pair_inflations[j][n] = get_pair_inflation_raw(j, n);
                i_inflation = min(i_inflation, pair_inflations[j][n]);
            }

            return i_inflation;
        }
        else
        {
            insert_into_spatial_hash(n);

            double ix = squares[n][0];
            double iy = squares[n][1];
            auto [cx, cy] = cell_indices(ix, iy);

            double cs = cell_size;
            double ax = (double) cx * cs - L;
            double by = (double) cy * cs - L;
            double d = min(min(ix - ax, ax + cs - ix), 
                           min(iy - by, by + cs - iy));

            temp_min_inflations.clear();
            temp_blames.clear();

            // squares that used to blame i
            for (int sq: guilt_sets[i])
            {
                auto [m_infl, inflicted_by] = find_min_inflation_spatial(sq, i, INF);
                temp_min_inflations.push_back(make_pair(sq, m_infl));
                i_inflation = min(i_inflation, m_infl);
                temp_blames.push_back(make_pair(sq, inflicted_by));
            }

            // the new position itself
            {
                auto [m_infl, inflicted_by] = find_min_inflation_spatial(n, i, INF);
                i_inflation = min(i_inflation, m_infl);
                temp_min_inflations.push_back(make_pair(n, m_infl));
                temp_blames.push_back(make_pair(n, inflicted_by));
            }

            double largest_infl = -1;
            for (int j = 0; j < n; j++) {
                largest_infl = max(min_inflations[j], largest_infl);
            }

            // only squares within this radius of i may be inflicted
            int c_rad = (int) ceil((2.0*largest_infl - d) / cs);

            for(int j = cx - c_rad; j <= cx + c_rad; j++)
            {
                for(int k = cy - c_rad; k <= cy + c_rad; k++)
                {
                    if (j < 0 || j >= grid_dim || k < 0 || k >= grid_dim) 
                        continue;

                    for (int sq : grid[j][k]) {
                        if (sq == i || sq == n || accused[sq] == i) 
                            continue;

                        double dx = squares[sq][0] - squares[i][0];
                        double dy = squares[sq][1] - squares[i][1];
                        double dist2 = dx*dx + dy*dy;
                        if (dist2 > 4.0 * min_inflations[sq] * min_inflations[sq]) 
                            continue;

                        auto [m_infl, inflicted_by] = find_min_inflation_spatial(sq, i, min_inflations[sq]);
                        temp_min_inflations.push_back(make_pair(sq, m_infl));
                        i_inflation = min(i_inflation, m_infl);
                        temp_blames.push_back(make_pair(sq, inflicted_by));
                    }
                }
            }

            return i_inflation;
        }
    }

    void accept_replacement(int i)
    {
        if (use_spatial_hash)
            remove_from_spatial_hash(i);

        squares[i][0] = squares[n][0];
        squares[i][1] = squares[n][1];
        squares[i][2] = squares[n][2];
        cos_table[i] = cos_table[n];
        sin_table[i] = sin_table[n];
        boundary_inflations[i] = boundary_inflations[n];

        if (!use_spatial_hash)
        {
            for (int j = 0; j < i; j++)
                pair_inflations[j][i] = pair_inflations[j][n];
            for (int j = i + 1; j < n; j++)
                pair_inflations[i][j] = pair_inflations[j][n];

            update_max_inflation();
        }
        else
        {
            insert_into_spatial_hash(i);
            guilt_sets[i].clear();

            for (auto &p : temp_min_inflations)
            {
                int sq = p.first;
                double new_mi = p.second;

                if (sq == n) {
                    // scratch => i
                    min_inflations[i] = new_mi;
                } else {
                    min_inflations[sq] = new_mi;
                }
            }

            for (auto &b : temp_blames)
            {
                int sq = b.first;
                int inflicted_by = b.second;

                if (sq == n)
                {
                    // new i
                    if (accused[i] >= 0) 
                        guilt_sets[accused[i]].erase(i);
                    accused[i] = inflicted_by;
                    if (inflicted_by >= 0) 
                        guilt_sets[inflicted_by].insert(i);
                }
                else if (inflicted_by == n)
                {
                    // sq now inflicted by i
                    if (accused[sq] >= 0) 
                        guilt_sets[accused[sq]].erase(sq);
                    accused[sq] = i;
                    guilt_sets[i].insert(sq);
                }
                else if (accused[sq] == i)
                {
                    // was inflicted by i, now inflicted_by
                    accused[sq] = inflicted_by;
                    if (inflicted_by >= 0)
                        guilt_sets[inflicted_by].insert(sq);
                }
            }

            temp_min_inflations.clear();
            temp_blames.clear();

            max_inflation = INF;
            for (int s = 0; s < n; s++) {
                max_inflation = min(max_inflation, min_inflations[s]);
            }
        }
    }

    void update_max_inflation()
    {
        max_inflation = INF;

        for (int i = 0; i < n; i++)
        {
            max_inflation = min(max_inflation, boundary_inflations[i]);
            for (int j = i + 1; j < n; j++)
            {
                max_inflation = min(max_inflation, pair_inflations[i][j]);
            }
        }
    }

    void log_state(const string& filename, int precision = 15)
    {
        ofstream logfile(filename, ios::app);
        if (!logfile.is_open()) return;

        logfile << fixed << setprecision(precision);
        logfile << "{";
        logfile << "\"L\":" << L << ",";
        logfile << "\"max_inflation\":" << max_inflation << ",";
        logfile << "\"squares\":[";
        for (int i = 0; i < n; ++i) {
            logfile << "[" << squares[i][0] << ","
                    << squares[i][1] << ","
                    << squares[i][2] << "]"
                    << (i < n-1 ? "," : "");
        }
        logfile << "]}\n";
        logfile.close();
    }

    static Configuration load_from_logfile(const string& filename)
    {
        ifstream logfile(filename);
        if (!logfile.is_open()) {
            cerr << "Error: Could not open logfile: " << filename << endl;
            exit(1);
        }
        
        string last_line;
        string line;
        while (getline(logfile, line)) {
            if (!line.empty()) {
                last_line = line;
            }
        }
        logfile.close();
        
        if (last_line.empty()) {
            cerr << "Error: Logfile is empty: " << filename << endl;
            exit(1);
        }
        
        size_t L_pos = last_line.find("\"L\":");
        size_t max_inflation_pos = last_line.find("\"max_inflation\":");
        size_t squares_pos = last_line.find("\"squares\":[");
        
        if (L_pos == string::npos || max_inflation_pos == string::npos || squares_pos == string::npos) {
            cerr << "Error: Invalid JSON format in logfile: " << filename << endl;
            exit(1);
        }
        
        // Extract L
        L_pos += 4; // skip "\"L\":"
        size_t L_end = last_line.find(",", L_pos);
        double L = stod(last_line.substr(L_pos, L_end - L_pos));
        
        // Extract squares array
        squares_pos += 11; // skip "\"squares\":["
        size_t squares_end = last_line.find("]}", squares_pos);
        string squares_str = last_line.substr(squares_pos, squares_end - squares_pos);
        
        // Count squares by counting ']'
        int n = 0;
        size_t pos = 0;
        while ((pos = squares_str.find(']', pos)) != string::npos) {
            n++;
            pos++;
        }
        
        Configuration config(n, L);
        config.squares.clear();
        config.cos_table.clear();
        config.sin_table.clear();
        
        // Parse each square
        string::size_type start = 0;
        for (int i = 0; i < n; i++) {
            size_t open_bracket = squares_str.find('[', start);
            size_t close_bracket = squares_str.find(']', open_bracket);
            if (open_bracket == string::npos || close_bracket == string::npos) {
                cerr << "Error: Invalid square format." << endl;
                exit(1);
            }
            
            string square_str = squares_str.substr(open_bracket + 1, close_bracket - open_bracket - 1);
            istringstream ss(square_str);
            string token;
            vector<double> values;
            while (getline(ss, token, ',')) {
                values.push_back(stod(token));
            }
            
            if (values.size() != 3) {
                cerr << "Error: Invalid square format (expected 3 values)." << endl;
                exit(1);
            }
            
            config.squares.push_back({values[0], values[1], values[2]});
            config.cos_table.push_back(cos(values[2]));
            config.sin_table.push_back(sin(values[2]));
            
            start = close_bracket + 1;
        }
        
        // add the scratchpad square
        config.squares.push_back({0.0, 0.0, 0.0});
        config.cos_table.push_back(1.0);
        config.sin_table.push_back(0.0);
        
        config.update_inflations_full();
        
        return config;
    }
};

void random_walking(Configuration& C, int num_attempts, double epsilon)
{
    double pos_eps = epsilon*(2*C.L);
    double angle_eps = epsilon*(PI/2);

    for (int k = 0; k < C.n * num_attempts; k++)
    {
        int i = k % C.n;

        double left_bound   = max(-C.L, C.squares[i][0] - pos_eps);
        double right_bound  = min(C.L,  C.squares[i][0] + pos_eps);
        double bottom_bound = max(-C.L, C.squares[i][1] - pos_eps);
        double top_bound    = min(C.L,  C.squares[i][1] + pos_eps);

        uniform_real_distribution<> x_dist(left_bound, right_bound);
        uniform_real_distribution<> y_dist(bottom_bound, top_bound);
        uniform_real_distribution<> angle_dist(C.squares[i][2] - angle_eps,
                C.squares[i][2] + angle_eps);

        double new_x = x_dist(gen);
        double new_y = y_dist(gen);
        double new_angle = angle_dist(gen);
        new_angle = fmod(new_angle, 2 * PI);
        if (new_angle < 0) new_angle += 2 * PI;

        double new_inflation = C.propose_replacement(i, new_x, new_y, new_angle);
        if (new_inflation >= C.max_inflation)
            C.accept_replacement(i);
    }
}

void billiard_of_squares(Configuration& C, double eps_init, double eps_min,
        double eps_max, int num_attempts,
        const string& logfile)
{
    bool log = !logfile.empty();
    double epsilon = eps_init;

    if (log)
        C.log_state(logfile);

    while (epsilon > eps_min)
    {
        double old_inflation = C.max_inflation;
        random_walking(C, num_attempts, epsilon);

        if (log)
            C.log_state(logfile);

        if (C.max_inflation > old_inflation)
            epsilon = min(epsilon * 2, eps_max);
        else
            epsilon = max(epsilon / 2, eps_min);
    }
}

void perturbation(Configuration& C, double epsilon)
{
    double pos_eps = epsilon*(2*C.L);
    double angle_eps = epsilon*(PI/2);

    for (int i = 0; i < C.n; i++)
    {
        double left_bound   = max(-C.L, C.squares[i][0] - pos_eps);
        double right_bound  = min(C.L,  C.squares[i][0] + pos_eps);
        double bottom_bound = max(-C.L, C.squares[i][1] - pos_eps);
        double top_bound    = min(C.L,  C.squares[i][1] + pos_eps);

        uniform_real_distribution<> x_dist(left_bound, right_bound);
        uniform_real_distribution<> y_dist(bottom_bound, top_bound);
        uniform_real_distribution<> angle_dist(C.squares[i][2] - angle_eps,
                C.squares[i][2] + angle_eps);

        double new_angle = angle_dist(gen);
        new_angle = fmod(new_angle, 2 * PI);
        if (new_angle < 0) new_angle += 2 * PI;

        C.set_square(i, x_dist(gen), y_dist(gen), new_angle);
    }

    C.update_inflations_full();
}

void perturb_and_billiards(Configuration& C, double eps_init, double eps_min, double eps_max, double factor, int num_attempts, string logfile)
{
    bool log = !logfile.empty();
    double epsilon = eps_init;

    if (log)
        C.log_state(logfile);

    while (epsilon > eps_min) {
        Configuration C_copy(C);

        perturbation(C_copy, epsilon);
        billiard_of_squares(C_copy, epsilon, max(epsilon/factor, eps_min), eps_max, num_attempts, "");

        if (C_copy.max_inflation > C.max_inflation) {
            C = C_copy;
            epsilon = min(epsilon * 2, eps_max);
        } else {
            epsilon = max(epsilon / 2, eps_min);
        }

        if (log)
            C.log_state(logfile);
    }
}

struct RunInfo {
    int run_id;
    double max_inflation;
    string logfile;

    RunInfo() : run_id(-1), max_inflation(0.0), logfile("") {}
    
    RunInfo(int id, double infl, const string& file) 
        : run_id(id), max_inflation(infl), logfile(file) {}
    
    // For sorting by max_inflation (descending)
    bool operator<(const RunInfo& other) const {
        return max_inflation > other.max_inflation;
    }
};

// Structure to track the best configurations
struct RunTracker {
    
    vector<RunInfo> best_runs;
    int top_k;
    
    RunTracker(int k = 10) : top_k(k) {}
    
    void add_run(int run_id, double max_inflation, const string& logfile) {
        best_runs.emplace_back(RunInfo(run_id, max_inflation, logfile));
        
        // Keep only the top_k best runs
        if (best_runs.size() > top_k) {
            sort(best_runs.begin(), best_runs.end());
            best_runs.resize(top_k);
        }
    }
    
    void print_best_runs() {
        sort(best_runs.begin(), best_runs.end());
        cout << "\n===== TOP " << best_runs.size() << " RUNS BY MAX INFLATION =====\n";
        for (size_t i = 0; i < best_runs.size(); i++) {
            cout << (i+1) << ". Run " << best_runs[i].run_id 
                 << ": max_inflation = " << best_runs[i].max_inflation 
                 << " (logfile: " << best_runs[i].logfile << ")\n";
        }
        cout << "=========================================\n\n";
    }
    
    void save_report(const string& filename) {
        ofstream report(filename);
        if (!report.is_open()) {
            cerr << "Error: Could not open report file: " << filename << endl;
            return;
        }
        
        sort(best_runs.begin(), best_runs.end());
        report << "# Top " << best_runs.size() << " Runs by Max Inflation\n\n";
        report << "| Rank | Run ID | Max Inflation | Logfile |\n";
        report << "|------|--------|---------------|---------|\n";
        
        for (size_t i = 0; i < best_runs.size(); i++) {
            report << "| " << (i+1) << " | " << best_runs[i].run_id 
                   << " | " << best_runs[i].max_inflation 
                   << " | " << best_runs[i].logfile << " |\n";
        }
        
        report.close();
        cout << "Report saved to " << filename << endl;
    }
    
    // Get the best run's logfile
    string get_best_logfile() {
        if (best_runs.empty()) return "";
        sort(best_runs.begin(), best_runs.end());
        return best_runs[0].logfile;
    }
};

void preliminary_billiards(int n, int num_runs, int top_k) {
    // Create output directory for log files
    string log_dir = "billiard_logs_" + to_string(n);
    std::filesystem::create_directory(log_dir);
    
    // Tracker for best runs
    RunTracker tracker(top_k);

    std::mutex io_mutex;
    std::mutex tracker_mutex;

    // Create a report directory
    string report_dir = "inflation_reports";
    std::filesystem::create_directory(report_dir);
    
    // Vector of futures to join
    vector<future<void>> futures;
    ThreadPool& pool = getGlobalThreadPool();

    for (int run = 0; run < num_runs; run++) {
        futures.push_back(pool.enqueue([&, run]() {
            // Collect messages locally
            ostringstream out;
            out << "Starting run " << run << "...\n";

            // Create a unique log file for this run
            string logfile = log_dir + "/billiard_run_" + to_string(run) + ".log";
            ofstream(logfile, ios::trunc).close(); // clear file
            
            // Run billiard_of_squares with a fresh configuration
            Configuration current_config(n, 1.0);
            billiard_of_squares(current_config, 0.5, 1e-8, 1.0, 10, logfile);
            
            out << "Run " << run << ": Completed with max_inflation: " 
                << current_config.max_inflation << "\n";
            
            // Update tracker under lock
            {
                lock_guard<mutex> lock(tracker_mutex);
                tracker.add_run(run, current_config.max_inflation, logfile);

                // Print best runs every 20 or at the last
                if ((run + 1) % 20 == 0 || run == num_runs - 1) {
                    tracker.print_best_runs();
                }
            }

            // Finally, print out local messages
            {
                lock_guard<mutex> lock(io_mutex);
                cout << out.str();
            }
        }));
    }

    // Wait for all runs to finish
    for (auto &f : futures) {
        f.get();
    }
    
    // Final report after all runs
    {
        lock_guard<mutex> lock(tracker_mutex);
        cout << "\n===== OPTIMIZATION COMPLETE =====\n";
        tracker.print_best_runs();
        
        string final_report = report_dir + "/final_report_" + to_string(n) + ".md";
        tracker.save_report(final_report);
        cout << "Final report saved to: " << final_report << endl;
    }
}

// Not parallelized, since it's used to fine tune a single configuration
void fine_tune_perturb(const string& logfile, int num_iterations, double factor) {
    // Create output directory for fine-tuned configurations
    string fine_tune_dir = "fine_tuned_configs";
    std::filesystem::create_directory(fine_tune_dir);
    
    cout << "Loading configuration from: " << logfile << endl;
    Configuration config = Configuration::load_from_logfile(logfile);
    cout << "Loaded configuration with max_inflation: " << config.max_inflation << endl;
    
    int n = config.n;
    double initial_inflation = config.max_inflation;
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    string timestamp = to_string(time);
    
    string base_filename = logfile.substr(logfile.find_last_of("/\\") + 1);
    string output_file = fine_tune_dir + "/fine_tuned_" + base_filename + "_" + timestamp + ".log";
    ofstream(output_file, ios::trunc).close(); // clear file
    
    cout << "Starting fine-tuning with perturbation..." << endl;
    cout << "Initial max_inflation: " << config.max_inflation << endl;
    
    perturb_and_billiards(config, 0.5, 1e-12, 1.0, factor, num_iterations, output_file);
    
    cout << "Fine-tuning complete!" << endl;
    cout << "Final max_inflation: " << config.max_inflation << endl;
    cout << "Improvement: " << (config.max_inflation - initial_inflation) << endl;
    cout << "Results saved to: " << output_file << endl;
}

void simulated_annealing(Configuration& C,
                         double T_init,
                         double T_min,
                         double cooling_rate,
                         int n_multiplier,
                         double target_acceptance_ratio,
                         double adjustment_factor,
                         const string& logfile)
{
    bool log = !logfile.empty();
    double T = T_init;
    
    uniform_real_distribution<> prob_dist(0.0, 1.0);

    double move_size_factor = 1.0;

    double best_inflation = C.max_inflation;
    Configuration best_config(C);

    if (log)
        C.log_state(logfile);

    vector<int> indices(C.n);
    iota(indices.begin(), indices.end(), 0);

    while (T > T_min)
    {
        int accepted_moves = 0;
        int total_moves = n_multiplier * C.n;

        for (int pass = 0; pass < n_multiplier; pass++)
        {
            shuffle(indices.begin(), indices.end(), gen);

            for (int idx = 0; idx < C.n; idx++)
            {
                int i = indices[idx];

                double pos_eps = move_size_factor * (2 * C.L);
                double angle_eps = move_size_factor * (PI / 2);

                double left_bound   = max(-C.L, C.squares[i][0] - pos_eps);
                double right_bound  = min(C.L,  C.squares[i][0] + pos_eps);
                double bottom_bound = max(-C.L, C.squares[i][1] - pos_eps);
                double top_bound    = min(C.L,  C.squares[i][1] + pos_eps);

                uniform_real_distribution<> x_dist(left_bound, right_bound);
                uniform_real_distribution<> y_dist(bottom_bound, top_bound);
                uniform_real_distribution<> angle_dist(C.squares[i][2] - angle_eps,
                                                      C.squares[i][2] + angle_eps);

                double new_x = x_dist(gen);
                double new_y = y_dist(gen);

                double new_angle = angle_dist(gen);
                new_angle = fmod(new_angle, 2 * PI);
                if (new_angle < 0) new_angle += 2 * PI;

                double new_inflation = C.propose_replacement(i, new_x, new_y, new_angle);

                bool accept = false;
                if (new_inflation >= C.max_inflation) {
                    // Always accept improvement
                    accept = true;
                } else {
                    double delta = new_inflation - C.max_inflation;
                    double acceptance_prob = exp(delta / T);
                    if (prob_dist(gen) < acceptance_prob)
                        accept = true;
                }

                if (accept) {
                    C.accept_replacement(i);
                    accepted_moves++;
                    if (C.max_inflation > best_inflation) {
                        best_inflation = C.max_inflation;
                        best_config = C;
                    }
                }
            }

            if (log)
                C.log_state(logfile);
        }

        double acceptance_ratio = (double)accepted_moves / total_moves;

        if (acceptance_ratio > target_acceptance_ratio) {
            move_size_factor *= adjustment_factor;
        }
        else if (acceptance_ratio < target_acceptance_ratio) {
            move_size_factor /= adjustment_factor;
        }

        T *= cooling_rate;
    }

    if (C.max_inflation < best_inflation) {
        C = best_config;
        if (log)
            C.log_state(logfile);
    }
}

struct SAParams {
    double T_init;
    double T_min;
    double cooling_rate;
    int n_multiplier;
    double target_acceptance_ratio;
    double adjustment_factor;
    
    SAParams(double t_init = 1.0, double t_min = 1e-8, double cr = 0.95, 
             int multiplier = 5, double tar = 0.3, double af = 1.1)
        : T_init(t_init), T_min(t_min), cooling_rate(cr),
          n_multiplier(multiplier),
          target_acceptance_ratio(tar), adjustment_factor(af) {}
};

void run_multi_cycle_simulated_annealing(int n, int num_runs, int top_k, 
                                        const vector<SAParams>& cycle_params) {
    string log_dir = "sa_logs_" + to_string(n);
    std::filesystem::create_directory(log_dir);
    
    RunTracker tracker(top_k);
    string report_dir = "inflation_reports";
    std::filesystem::create_directory(report_dir);

    // Same two mutex approach
    std::mutex io_mutex;
    std::mutex tracker_mutex;

    ThreadPool& pool = getGlobalThreadPool();
    vector<future<void>> futures;

    for (int run = 0; run < num_runs; run++) {
        futures.push_back(pool.enqueue([&, run]() {
            ostringstream out;
            out << "Run " << run << " starting...\n";

            // One main logfile for the entire run
            string main_logfile = log_dir + "/sa_run_" + to_string(run) + ".log";
            ofstream(main_logfile, ios::trunc).close(); // clear file

            // Start with a fresh config
            Configuration current_config(n, 1.0);
            out << "Run " << run << ", Cycle 0"
                << ": initial max_inflation=" << current_config.max_inflation << "\n";

            // Do each cycle
            for (size_t cycle = 0; cycle < cycle_params.size(); cycle++) {
                const SAParams& params = cycle_params[cycle];

                // A separate log for each cycle
                string cycle_logfile = log_dir + "/sa_run_" + to_string(run)
                                     + "_cycle_" + to_string(cycle) + ".log";
                ofstream(cycle_logfile, ios::trunc).close();

                int multiplier = params.n_multiplier;
                if (multiplier <= 0) {
                    multiplier = 5;
                }

                simulated_annealing(current_config,
                                    params.T_init,
                                    params.T_min,
                                    params.cooling_rate,
                                    multiplier,
                                    params.target_acceptance_ratio,
                                    params.adjustment_factor,
                                    cycle_logfile);

                out << "Run " << run << ", Cycle " << cycle 
                    << ": Completed with max_inflation=" 
                    << current_config.max_inflation << "\n";

                // Append cycle's log to main log
                {
                    ifstream cycle_in(cycle_logfile);
                    ofstream main_out(main_logfile, ios::app);
                    if (cycle_in && main_out) {
                        main_out << cycle_in.rdbuf();
                    } else {
                        out << "Warning: Could not append cycle " << cycle
                            << " log to main log.\n";
                    }
                }
            }

            // Update tracker
            {
                lock_guard<mutex> lock(tracker_mutex);
                tracker.add_run(run, current_config.max_inflation, main_logfile);

                // Print best runs every 10 or final
                if ((run + 1) % 10 == 0 || run == num_runs - 1) {
                    tracker.print_best_runs();
                }
            }

            // Output everything for this run
            {
                lock_guard<mutex> lock(io_mutex);
                cout << out.str();
            }
        }));
    }

    for (auto &f : futures) {
        f.get();
    }

    {
        lock_guard<mutex> lock(tracker_mutex);
        cout << "\n===== OPTIMIZATION COMPLETE =====\n";
        tracker.print_best_runs();

        string final_report = report_dir + "/sa_report_" + to_string(n) + ".md";
        tracker.save_report(final_report);
        cout << "Final report saved to: " << final_report << endl;
    }
}

SAParams parse_sa_cycle_params(int argc, char* argv[], int& arg_index, int n) {
    SAParams params;
    
    if (arg_index < argc) params.T_init = stod(argv[arg_index++]);
    if (arg_index < argc) params.T_min = stod(argv[arg_index++]);
    if (arg_index < argc) params.cooling_rate = stod(argv[arg_index++]);
    if (arg_index < argc) params.n_multiplier = stoi(argv[arg_index++]);
    else params.n_multiplier = 5; 
    if (arg_index < argc) params.target_acceptance_ratio = stod(argv[arg_index++]);
    if (arg_index < argc) params.adjustment_factor = stod(argv[arg_index++]);
    
    return params;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage:" << endl;
        cerr << "  Preliminary search: " << argv[0] << " <num_squares> [num_runs=1000] [top_k=10]\n";
        cerr << "  Fine-tuning: " << argv[0] << " --tune <logfile> [num_iterations=10] [factor=1.5]\n";
        cerr << "  Single cycle SA: " << argv[0]
             << " --sa <num_squares> [num_runs=100] [top_k=10] [T_init=1.0]"
             << " [T_min=1e-8] [cooling_rate=0.95] [n_multiplier=5]"
             << " [target_acc_ratio=0.3] [adjustment_factor=1.1]\n";
        cerr << "  Multi-cycle SA: " << argv[0]
             << " --multi-sa <num_squares> <num_cycles> [num_runs=100] [top_k=10]\n"
             << "             [cycle1_T_init] [cycle1_T_min] [cycle1_cooling] [cycle1_multiplier]\n"
             << "             [cycle1_acc_ratio] [cycle1_adj_factor] [cycle2_T_init] ...\n";
        cerr << "    Each cycle's parameters are optional and use defaults if not specified.\n";
        return 1;
    }

    string first_arg = argv[1];
    if (first_arg == "--tune" || first_arg == "-t") {
        if (argc < 3) {
            cerr << "Error: Fine-tuning mode requires a logfile.\n";
            cerr << "Usage: " << argv[0] << " --tune <logfile> [num_iterations=10] [factor=1.5]\n";
            return 1;
        }
        
        string logfile = argv[2];
        int num_iterations = (argc > 3) ? stoi(argv[3]) : 10;
        double factor = (argc > 4) ? stod(argv[4]) : 1.5;
        fine_tune_perturb(logfile, num_iterations, factor);
    }
    else if (first_arg == "--sa") {
        if (argc < 3) {
            cerr << "Error: Simulated annealing mode requires number of squares.\n";
            return 1;
        }
        
        int n = stoi(argv[2]);
        int num_runs = (argc > 3) ? stoi(argv[3]) : 100;
        int top_k = (argc > 4) ? stoi(argv[4]) : 10;
        
        double T_init = (argc > 5) ? stod(argv[5]) : 1.0;
        double T_min = (argc > 6) ? stod(argv[6]) : 1e-8;
        double cooling_rate = (argc > 7) ? stod(argv[7]) : 0.95;
        int n_multiplier = (argc > 8) ? stoi(argv[8]) : 5;
        double target_acceptance_ratio = (argc > 9) ? stod(argv[9]) : 0.3;
        double adjustment_factor = (argc > 10) ? stod(argv[10]) : 1.1;
        
        vector<SAParams> cycle_params = {
            SAParams(T_init, T_min, cooling_rate, n_multiplier,
                     target_acceptance_ratio, adjustment_factor)
        };
        
        run_multi_cycle_simulated_annealing(n, num_runs, top_k, cycle_params);
    }
    else if (first_arg == "--multi-sa") {
        if (argc < 4) {
            cerr << "Error: Multi-cycle SA mode requires <num_squares> <num_cycles>.\n";
            return 1;
        }
        
        int n = stoi(argv[2]);
        int num_cycles = stoi(argv[3]);
        int num_runs = (argc > 4) ? stoi(argv[4]) : 100;
        int top_k = (argc > 5) ? stoi(argv[5]) : 10;
        
        vector<SAParams> cycle_params;
        int arg_index = 6;
        
        for (int cycle = 0; cycle < num_cycles; cycle++) {
            if (arg_index >= argc) {
                cycle_params.push_back(SAParams(1.0, 1e-8, 0.95, 5, 0.3, 1.1));
            } else {
                cycle_params.push_back(parse_sa_cycle_params(argc, argv, arg_index, n));
            }
        }
        
        run_multi_cycle_simulated_annealing(n, num_runs, top_k, cycle_params);
    }
    else {
        // Preliminary billiards mode
        int n = stoi(argv[1]);
        int num_runs = (argc > 2) ? stoi(argv[2]) : 1000;
        int top_k = (argc > 3) ? stoi(argv[3]) : 10;
        
        preliminary_billiards(n, num_runs, top_k);
    }

    return 0;
}

