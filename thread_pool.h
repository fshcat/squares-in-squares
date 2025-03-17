#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <atomic>
#include <iostream>

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;

public:
    // Constructor: spawn N worker threads
    explicit ThreadPool(size_t numThreads) : stop_(false) {
        workers_.reserve(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        // Wait until we have a task or we are stopping
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) 
                            return;  // all done

                        // Pop the next task
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    // Run the task outside the lock
                    task();
                }
            });
        }
    }

    // Destructor: join all threads
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_) {
            w.join();
        }
    }

    // Enqueue a single function to run, return immediately 
    void enqueue(std::function<void()> f) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.push(std::move(f));
        }
        cv_.notify_one();  
    }

    // Run func(i) for i in [start, end) and block until all calls to func() have completed.
    template <typename F>
    void parallel_for(int start, int end, F func) {
        if (start >= end) return;

        // Launch one loop per worker, with an index keeping track of the current task for each worker
        const size_t num_workers = workers_.size();
        auto current_index = std::make_shared<std::atomic<int>>(start);

        // A promise/future pair to know when all tasks have finished
        auto barrier = std::make_shared<std::promise<void>>();
        auto barrier_future = barrier->get_future();

        // How many tasks remain to finish before we can set the barrier
        auto active_count = std::make_shared<std::atomic<size_t>>(num_workers);

        // Each task runs in a worker thread until all indices in [start, end) are done
        auto loopBody = [=]() {
            while (true) {
                int i = current_index->fetch_add(1);
                if (i >= end) break;
                func(i);
            }

            if (active_count->fetch_sub(1) == 1)
                barrier->set_value();
        };

        // Dispatch worker loops
        for (size_t w = 0; w < num_workers; w++)
            enqueue(loopBody);

        // Block until the last worker sets the promise
        barrier_future.wait();
    }
};

#endif // THREADPOOL_H
