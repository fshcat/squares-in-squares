#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "thread_pool.h"

using namespace std;

static bool g_sequential = true;

class Configuration;
void log_squares(Configuration& C);

const double PI = M_PI;
const double INF = numeric_limits<double>::infinity();
const unsigned DEFAULT_WORKERS = 4;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

random_device rd;
mt19937 gen(rd());

ThreadPool& getGlobalThreadPool()
{
    static ThreadPool pool(std::max(DEFAULT_WORKERS, thread::hardware_concurrency()));
    return pool;
}

// Structure to represent a configuration of squares
class Configuration {
    public:
        int n;
        double L;
        vector<array<double, 3>> squares;
        vector<double> cos_table;
        vector<double> sin_table;

        vector<vector<double>> pair_inflations;
        vector<double> boundary_inflations;
        double max_inflation;

        vector<pair<int,int>> inflation_pairs;

        Configuration(int num_squares, double length)
            : n(num_squares),
            L(length),
            pair_inflations(num_squares+1, vector<double>(num_squares+1, 0.0)),
            boundary_inflations(num_squares+1, 0),
            max_inflation(INF)
        {
            inflation_pairs.reserve(n*(n-1)/2);
            for (int i = 0; i < n; ++i) 
                for (int j = i+1; j < n; ++j) 
                    inflation_pairs.push_back({i, j});

            // Initialize squares to random orientations
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

            // Calculate pairwise and boundary inflation values
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
            inflation_pairs(other.inflation_pairs) 
        {}

        void set_square(int i, double a, double b, double theta)
        {
            squares[i][0] = a;
            squares[i][1] = b;
            squares[i][2] = theta;
            cos_table[i] = cos(theta);
            sin_table[i] = sin(theta);
        }

        // Calculates the maximum inflation of a pair of squares, assuming
        // one of the squares is at position (0, 0) with angle 0. The other square
        // is as given in the parameters
        double psi_0(double a, double b, double theta)
        {
            double min_val = INF;

            for (int i = 0; i < 4; i++)
            {
                double angle = theta + i*(PI/2) + PI/4;

                double numer = abs(a) + abs(b);
                double denom = abs(1 - sqrt(2)*sgn(a*b)*sin(angle));

                min_val = min(min_val, numer / denom);
            }

            return min_val;
        }

        // Calculates the maximum inflation of an arbitrary pair of squares,
        // given by index
        double get_pair_inflation(int i, int j)
        {
            double a = (squares[j][0] - squares[i][0]) * cos_table[i]
                + (squares[j][1] - squares[i][1]) * sin_table[i];
            double b = -(squares[j][0] - squares[i][0]) * sin_table[i]
                + (squares[j][1] - squares[i][1]) * cos_table[i];
            double t = squares[j][2] - squares[i][2];

            double psi_1 = psi_0(a, b, t);

            a = (squares[i][0] - squares[j][0]) * cos_table[j]
                + (squares[i][1] - squares[j][1]) * sin_table[j];
            b = -(squares[i][0] - squares[j][0]) * sin_table[j]
                + (squares[i][1] - squares[j][1]) * cos_table[j];
            t = squares[i][2] - squares[j][2];

            double psi_2 = psi_0(a, b, t);

            return max(psi_1, psi_2);
        }

        // Calculate the maximal inflation of a square against the bounding box
        double get_boundary_inflation(int i)
        {
            double numer = L - max(abs(squares[i][0]), abs(squares[i][1]));
            double denom = max(abs(cos_table[i]), abs(sin_table[i]));
            return numer / denom;
        }

        // Recalculate the pairwise inflations for all squares in parallel
        void update_all_pair_inflations()
        {

            if (!g_sequential) {
                // parallel path
                ThreadPool& pool = getGlobalThreadPool();
                pool.parallel_for(0, (int)inflation_pairs.size(), [&](int idx){
                        auto [i, j] = inflation_pairs[idx];
                        pair_inflations[i][j] = get_pair_inflation(i, j);
                        });
            } else {
                // sequential path
                for (int idx = 0; idx < (int)inflation_pairs.size(); idx++) {
                    auto [i, j] = inflation_pairs[idx];
                    pair_inflations[i][j] = get_pair_inflation(i, j);
                }
            }

            // Reduce over pairwise inflations
            for (auto &pr : inflation_pairs) {
                int i = pr.first;
                int j = pr.second;
                max_inflation = min(max_inflation, pair_inflations[i][j]);
            }
        }

        // Recalculate the inflations for all squares against the boundary in parallel
        void update_all_boundary_inflations()
        {
            ThreadPool& pool = getGlobalThreadPool();

            if (!g_sequential) {
                ThreadPool& pool = getGlobalThreadPool();
                pool.parallel_for(0, n, [&](int i){
                        boundary_inflations[i] = get_boundary_inflation(i);
                        });
            } else {
                // sequential
                for (int i = 0; i < n; i++) {
                    boundary_inflations[i] = get_boundary_inflation(i);
                }
            }


            // Reduction over boundary inflations
            for (int i = 0; i < n; ++i) 
                max_inflation = std::min(max_inflation, boundary_inflations[i]);
        }

        // Recalculate all inflations
        void update_inflations_full()
        {
            // Reset max_inflation
            max_inflation = INF;

            update_all_pair_inflations();
            update_all_boundary_inflations();
        }

        // Get inflation for a proposed square replacing square i, in parallel
        double propose_replacement(int i, double a, double b, double theta)
        {
            double i_inflation = INF;

            // Set the scratchpad square
            set_square(n, a, b, theta);

            // Get inflation of square against boundary
            double bound_inflation = get_boundary_inflation(n);
            boundary_inflations[n] = bound_inflation;
            i_inflation = min(i_inflation, bound_inflation);

            if (!g_sequential) {
                // parallel version
                ThreadPool& pool = getGlobalThreadPool();
                pool.parallel_for(0, n-1, [&](int idx){
                        int j = (idx >= i) ? idx+1 : idx;
                        pair_inflations[j][n] = get_pair_inflation(j, n);
                        });
            } else {
                // sequential version
                for (int idx = 0; idx < n-1; idx++) {
                    int j = (idx >= i) ? idx+1 : idx;
                    pair_inflations[j][n] = get_pair_inflation(j, n);
                }
            }

            for (int idx = 0; idx < n-1; idx++) {
                int j = (idx >= i) ? idx+1 : idx;
                i_inflation = min(i_inflation, pair_inflations[j][n]);
            }

            return i_inflation;

        }

        void accept_replacement(int i)
        {
            squares[i][0] = squares[n][0];
            squares[i][1] = squares[n][1];
            squares[i][2] = squares[n][2];
            cos_table[i] = cos_table[n];
            sin_table[i] = sin_table[n];

            for (int j = 0; j < i; j++)
                pair_inflations[j][i] = pair_inflations[j][n];

            for (int j = i + 1; j < n; j++)
                pair_inflations[i][j] = pair_inflations[j][n];

            boundary_inflations[i] = boundary_inflations[n];

            update_max_inflation();
        }

        // Update max_inflation after accept_replacement modifies the tables
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

        // Logs state of the configuration as JSON
        void log_state(const string& filename)
        {
            ofstream logfile(filename, ios::app);
            if (!logfile.is_open()) return;

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
};

void random_walking(Configuration& C, int num_attempts, double epsilon)
{
    double pos_eps = epsilon*(2*C.L);
    double angle_eps = epsilon*(PI/2);

    for (int k = 0; k < num_attempts; k++)
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

        double new_angle = angle_dist(gen);
        new_angle = fmod(new_angle, 2 * PI);
        if (new_angle < 0) new_angle += 2 * PI;

        double new_inflation = C.propose_replacement(
                i, x_dist(gen), y_dist(gen), new_angle
                );
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
    C.update_max_inflation();
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
        billiard_of_squares(C_copy, epsilon, epsilon/factor, eps_max, num_attempts, "");

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

int main(int argc, char* argv[])
{
    int n = stoi(argv[1]); 
    double factor = stod(argv[2]);
    string logfile = argv[3];
    ofstream(logfile, ios::trunc).close(); // clear file

    double L = 1.0;
    Configuration init(n, L);
    init.log_state(logfile);

    billiard_of_squares(init, 0.5, 1e-8, 1.0, 1000, logfile);
    perturb_and_billiards(init, 0.5, 1e-12, 1.0, factor, 1000, logfile);

    return 0;
}
