//
// Copyright (c) Maginatics, Inc. All Rights Reserved.
// For more information, please see COPYRIGHT in the top-level directory.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_INCLUDE_THREADPOOL_DETAIL_WORKER_H_
#define SRC_INCLUDE_THREADPOOL_DETAIL_WORKER_H_

#include <maginatics/threadpool/detail/pool.h>

namespace maginatics {
namespace detail {

class Worker {
public:
    explicit Worker(Pool *pool);
    // Launch a thread and start the run loop
    bool start();
    // Join the thread
    void join();
private:
    void run(); // Main run loop
    Pool *pool_; // Pool handle
    cxx::thread thread_; // Execution thread
};

inline Worker::Worker(Pool *pool)
    : pool_(pool) {
    assert(pool_ != NULL);
}

inline void Worker::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

inline bool Worker::start() {
    assert(!thread_.joinable());
    thread_ = cxx::thread([this]() {
            run();
        });
    return thread_.joinable();
}

inline void Worker::run() {
    try {
        while (pool_->runTask(this)) { }
    } catch (...) {
        // Account for dead workers
        pool_->workerTerminatedUnexpectedly(this);
        // Re-throw
        throw;
    }
}

} // detail namespace
} // maginatics namespace

#endif // SRC_INCLUDE_THREADPOOL_DETAIL_WORKER_H_
