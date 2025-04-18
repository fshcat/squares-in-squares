#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <algorithm>

/**
 * Simple ThreadPool that:
 *  - Spawns 'threads' worker threads upon construction.
 *  - Exposes an 'enqueue' method to add tasks.
 *  - Waits on all workers to finish when destructed.
 *
 * Also has a 'parallel_for' convenience function to parallelize a for-loop
 * in the range [start, end).
 */
class ThreadPool {
public:
    // Create a thread pool with 'threads' worker threads
    explicit ThreadPool(size_t threads);

    // Enqueue any callable, returns a std::future<result_type>
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Simple parallel_for that splits the loop among the available threads
    void parallel_for(int start, int end, const std::function<void(int)>& func);

    // Destructor joins all threads
    ~ThreadPool();

    // Optional: how many workers in this pool?
    inline size_t size() const { return workers.size(); }

private:
    // Keep track of worker threads
    std::vector<std::thread> workers;

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};


// ------------------ Implementation ------------------

// Constructor: launch the requested number of workers
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
    if (threads == 0) {
        throw std::runtime_error("ThreadPool: number of threads cannot be zero.");
    }

    for(size_t i = 0; i < threads; ++i)
    {
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;
                    {   // acquire lock
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        // wait on condition until either:
                        //   1) there's a new task, or
                        //   2) we're stopping
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });

                        // If no tasks and we're stopping, break out
                        if(this->stop && this->tasks.empty())
                            return;

                        // Pop the next task
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    // Run the task
                    task();
                }
            }
        );
    }
}

// Enqueue a new task and return its future result
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    // Package the callable into a shared task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one(); // wake up one worker

    return res;
}

// parallel_for: distribute the loop [start, end) among the worker threads
inline void ThreadPool::parallel_for(int start, int end, const std::function<void(int)>& func)
{
    int length = end - start;
    if (length <= 0) return;

    // If we have no workers, just do it in the current thread
    size_t num_workers = workers.size();
    if (num_workers == 0)
    {
        for (int i = start; i < end; i++)
            func(i);
        return;
    }

    // Determine chunk size
    int chunk_size = (length + static_cast<int>(num_workers) - 1) / static_cast<int>(num_workers);
    std::vector<std::future<void>> futures;
    futures.reserve(num_workers);

    // Assign each chunk as one task
    int current = start;
    for (size_t t = 0; t < num_workers; t++)
    {
        int chunk_start = current;
        int chunk_end   = std::min(end, chunk_start + chunk_size);
        current = chunk_end;

        if (chunk_start >= end)
            break;

        // Enqueue this chunk
        futures.push_back(
            enqueue([chunk_start, chunk_end, &func]() {
                for (int i = chunk_start; i < chunk_end; i++) {
                    func(i);
                }
            })
        );
    }

    // Wait for all chunks to complete
    for (auto &f : futures) {
        f.get();
    }
}

// Destructor: signal all workers to stop, then join them
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all(); // wake all threads
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}


