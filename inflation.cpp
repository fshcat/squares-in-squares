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

using namespace std;

class Configuration;
void log_squares(Configuration& C);
const double PI = M_PI;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
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

    Configuration(int num_squares, double length)
        : n(num_squares),
          L(length),
          pair_inflations(num_squares+1, vector<double>(num_squares+1, 0.0)),
          boundary_inflations(num_squares+1, 0),
          max_inflation(INFINITY)
    {
        // Initialize squares to random orientations
        random_device rd;
        mt19937 gen(rd());
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

    // Calculates the maximum inflation of a pair of squares, assuming
    // one of the squares is at position (0, 0) with angle 0. The other square
    // is as given in the parameters
    double psi_0(double a, double b, double theta)
    {
        double min_val = INFINITY;

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

        // Build a work queue with all (i,j) tasks
        struct PairTask { int i, j; };
        queue<PairTask> work_queue;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                work_queue.push({i, j});
            }
        }

        // Prepare concurrency tools
        mutex queue_mutex;
        condition_variable cv;

        // Worker function to consume tasks
        auto worker = [&]() {
            while (true) {
                PairTask task;
                {
                    unique_lock<mutex> lock(queue_mutex);
                    if (work_queue.empty()) {
                        return; // no more tasks
                    }
                    task = work_queue.front();
                    work_queue.pop();
                }
                double inflation = get_pair_inflation(task.i, task.j);
                pair_inflations[task.i][task.j] = inflation;
            }
        };

        // Spawn a pool of threads
        unsigned int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // fallback
        vector<thread> threads;
        threads.reserve(num_threads);

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker);
        }

        // Join them
        for (auto &th : threads) {
            th.join();
        }

        // Reduce over the results
        double new_max_inflation = numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                new_max_inflation = min(new_max_inflation, pair_inflations[i][j]);
            }
        }
        
        // Merge with existing max_inflation
        max_inflation = min(max_inflation, new_max_inflation);
    }

    // Recalculate the inflations for all squares against the boundary in parallel
    void update_all_boundary_inflations()
    {
        // Similar approach, but tasks are single "i"
        queue<int> work_queue;
        for (int i = 0; i < n; ++i) {
            work_queue.push(i);
        }

        mutex queue_mutex;
        condition_variable cv;

        auto worker = [&]() {
            while (true) {
                int idx;
                {
                    unique_lock<mutex> lock(queue_mutex);
                    if (work_queue.empty()) {
                        return;
                    }
                    idx = work_queue.front();
                    work_queue.pop();
                }
                double inflation = get_boundary_inflation(idx);
                boundary_inflations[idx] = inflation;
            }
        };

        unsigned int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        vector<thread> threads;
        threads.reserve(num_threads);

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker);
        }

        for (auto &th : threads) {
            th.join();
        }

        // Reduce
        double new_max_inflation = numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            new_max_inflation = min(new_max_inflation, boundary_inflations[i]);
        }
        max_inflation = min(max_inflation, new_max_inflation);
    }

    // Recalculate all inflations
    void update_inflations_full()
    {
        // Reset max_inflation before we do the parallel updates
        max_inflation = INFINITY;

        update_all_pair_inflations();
        update_all_boundary_inflations();
    }

    // Get inflation for a proposed square replacing square i
    double propose_replacement(int i, double a, double b, double theta)
    {
        double i_inflation = INFINITY;

        // Set the scratchpad square
        squares[n][0] = a;
        squares[n][1] = b;
        squares[n][2] = theta;
        cos_table[n] = cos(theta);
        sin_table[n] = sin(theta);

        for (int j = 0; j < n; j++)
        {
            if (j != i)
            {
                double j_inf = get_pair_inflation(j, n);
                pair_inflations[j][n] = j_inf;
                i_inflation = min(i_inflation, j_inf);
            }
        }

        double bound_inflation = get_boundary_inflation(n);
        boundary_inflations[n] = bound_inflation;
        i_inflation = min(i_inflation, bound_inflation);

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
        max_inflation = INFINITY;

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
    random_device rd;
    mt19937 gen(rd());

    for (int k = 0; k < num_attempts; k++)
    {
        int i = k % C.n;

        double left_bound   = max(-C.L, C.squares[i][0] - epsilon);
        double right_bound  = min(C.L,  C.squares[i][0] + epsilon);
        double bottom_bound = max(-C.L, C.squares[i][1] - epsilon);
        double top_bound    = min(C.L,  C.squares[i][1] + epsilon);

        uniform_real_distribution<> x_dist(left_bound, right_bound);
        uniform_real_distribution<> y_dist(bottom_bound, top_bound);
        uniform_real_distribution<> angle_dist(C.squares[i][2] - epsilon,
                                              C.squares[i][2] + epsilon);

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
        ofstream(logfile, ios::trunc).close(); // clear file if logging

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

void log_squares(Configuration& C)
{
    for (int i = 0; i < C.n; i++)
    {
        cout << C.squares[i][0] << " " << C.squares[i][1] << " "
             << C.squares[i][2] << "\n";
    }
}

int main(int argc, char* argv[])
{
    int n = stoi(argv[1]); 
    string logfile = argv[2];

    double L = 1.0;
    Configuration init(n, L);

    billiard_of_squares(init, 0.1, 1e-8, 1.0, 1000, logfile);

    return 0;
}
