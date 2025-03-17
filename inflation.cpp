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
        void log_state(const string& filename, int precision = 15)
        {
            ofstream logfile(filename, ios::app);
            if (!logfile.is_open()) return;

            // Set precision for floating point values
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
            
            // Parse the JSON line
            size_t L_pos = last_line.find("\"L\":");
            size_t max_inflation_pos = last_line.find("\"max_inflation\":");
            size_t squares_pos = last_line.find("\"squares\":[");
            
            if (L_pos == string::npos || max_inflation_pos == string::npos || squares_pos == string::npos) {
                cerr << "Error: Invalid JSON format in logfile: " << filename << endl;
                exit(1);
            }
            
            // Extract L value
            L_pos += 4; // Move past "\"L\":
            size_t L_end = last_line.find(",", L_pos);
            double L = stod(last_line.substr(L_pos, L_end - L_pos));
            
            // Extract squares array
            squares_pos += 11; // Move past "\"squares\":["
            size_t squares_end = last_line.find("]}", squares_pos);
            string squares_str = last_line.substr(squares_pos, squares_end - squares_pos);
            
            // Count number of squares by counting occurrences of ']'
            int n = 0;
            size_t pos = 0;
            while ((pos = squares_str.find(']', pos)) != string::npos) {
                n++;
                pos++;
            }
            
            // Create configuration with correct n and L
            Configuration config(n, L);
            
            // Clear the randomly initialized squares
            config.squares.clear();
            config.cos_table.clear();
            config.sin_table.clear();
            
            // Parse each square
            string::size_type start = 0;
            for (int i = 0; i < n; i++) {
                size_t open_bracket = squares_str.find('[', start);
                size_t close_bracket = squares_str.find(']', open_bracket);
                
                if (open_bracket == string::npos || close_bracket == string::npos) {
                    cerr << "Error: Invalid square format in logfile: " << filename << endl;
                    exit(1);
                }
                
                string square_str = squares_str.substr(open_bracket + 1, close_bracket - open_bracket - 1);
                
                // Parse x, y, theta
                istringstream ss(square_str);
                string token;
                vector<double> values;
                
                while (getline(ss, token, ',')) {
                    values.push_back(stod(token));
                }
                
                if (values.size() != 3) {
                    cerr << "Error: Invalid square format (expected 3 values): " << square_str << endl;
                    exit(1);
                }
                
                // Add square to configuration
                config.squares.push_back({values[0], values[1], values[2]});
                config.cos_table.push_back(cos(values[2]));
                config.sin_table.push_back(sin(values[2]));
                
                start = close_bracket + 1;
            }
            
            // Add the scratchpad square
            config.squares.push_back({0.0, 0.0, 0.0});
            config.cos_table.push_back(1.0);
            config.sin_table.push_back(0.0);
            
            // Recalculate inflations
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
        report << "|------|--------|---------------|--------|\n";
        
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
    
    // Create a report directory
    string report_dir = "inflation_reports";
    std::filesystem::create_directory(report_dir);
    
    for (int run = 0; run < num_runs; run++) {
        // Create a unique log file for this run
        string logfile = log_dir + "/billiard_run_" + to_string(run) + ".log";
        ofstream(logfile, ios::trunc).close(); // clear file
        
        // Run billiard_of_squares with a fresh configuration
        Configuration current_config(n, 1.0);
        billiard_of_squares(current_config, 0.5, 1e-8, 1.0, 10, logfile);
        
        cout << "Run " << run << ": Completed with max_inflation: " 
             << current_config.max_inflation << endl;
             
        tracker.add_run(run, current_config.max_inflation, logfile);
        
        // Every 20 runs, print a report of the best runs
        if ((run + 1) % 20 == 0 || run == num_runs - 1)
            tracker.print_best_runs();
    }
    
    cout << "\n===== OPTIMIZATION COMPLETE =====\n";
    tracker.print_best_runs();
    
    // Save final report
    string final_report = report_dir + "/final_report_" + to_string(n) + ".md";
    tracker.save_report(final_report);
    cout << "Final report saved to: " << final_report << endl;
}

void fine_tune_perturb(const string& logfile, int num_iterations, double factor) {
    // Create output directory for fine-tuned configurations
    string fine_tune_dir = "fine_tuned_configs";
    std::filesystem::create_directory(fine_tune_dir);
    
    cout << "Loading configuration from: " << logfile << endl;
    Configuration config = Configuration::load_from_logfile(logfile);
    cout << "Loaded configuration with max_inflation: " << config.max_inflation << endl;
    
    // Extract configuration details for output filename
    int n = config.n;
    double initial_inflation = config.max_inflation;
    
    // Create output logfile name based on input file and timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    string timestamp = to_string(time);
    
    string base_filename = logfile.substr(logfile.find_last_of("/\\") + 1);
    string output_file = fine_tune_dir + "/fine_tuned_" + base_filename + "_" + timestamp + ".log";
    ofstream(output_file, ios::trunc).close(); // clear file
    
    cout << "Starting fine-tuning with perturbation..." << endl;
    cout << "Initial max_inflation: " << config.max_inflation << endl;
    
    // Run perturb_and_billiards to fine-tune the configuration
    perturb_and_billiards(config, 0.5, 1e-12, 1.0, factor, num_iterations, output_file);
    
    cout << "Fine-tuning complete!" << endl;
    cout << "Final max_inflation: " << config.max_inflation << endl;
    cout << "Improvement: " << (config.max_inflation - initial_inflation) << endl;
    cout << "Results saved to: " << output_file << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage:" << endl;
        cerr << "  Preliminary search: " << argv[0] << " <num_squares> [num_runs=1000] [top_k=10]" << endl;
        cerr << "  Fine-tuning: " << argv[0] << " --tune <logfile> [num_iterations=10] [factor=1.5]" << endl;
        return 1;
    }

    string first_arg = argv[1];
    if (first_arg == "--tune" || first_arg == "-t") {
        if (argc < 3) {
            cerr << "Error: Fine-tuning mode requires a logfile" << endl;
            cerr << "Usage: " << argv[0] << " --tune <logfile> [num_iterations=10] [factor=1.5]" << endl;
            return 1;
        }
        
        string logfile = argv[2];
        int num_iterations = (argc > 3) ? stoi(argv[3]) : 10;
        double factor = (argc > 4) ? stod(argv[4]) : 1.5;
        
        fine_tune_perturb(logfile, num_iterations, factor);
    }
    else {
        int n = stoi(argv[1]);
        
        int num_runs = (argc > 2) ? stoi(argv[2]) : 1000;
        int top_k = (argc > 3) ? stoi(argv[3]) : 10;
        
        preliminary_billiards(n, num_runs, top_k);
    }

    return 0;
}
