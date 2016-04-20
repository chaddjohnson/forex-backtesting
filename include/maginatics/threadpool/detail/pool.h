//
// Copyright (c) Maginatics, Inc. All Rights Reserved.
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

#ifndef MAGINATICS_THREADPOOL_DETAIL_POOL_H_
#define MAGINATICS_THREADPOOL_DETAIL_POOL_H_

#include <deque>
#include <vector>

#include <maginatics/threadpool/detail/port.h>

namespace maginatics {
namespace detail {

// Forward declarations
class Worker;

class Pool {
public:
    Pool(int64_t minPoolSize, int64_t maxPoolSize, int64_t keepAlive)
        : minPoolSize_(minPoolSize),
          maxPoolSize_(maxPoolSize),
          keepAlive_(keepAlive),
          poolSize_(0),
          activeWorkers_(0),
          shutdown_(false) {
        initPool();
    }

    ~Pool() {
        shutdown();
    }

    //
    // Public ThreadPool implementation
    //

    bool execute(cxx::function<void()> const& task);
    template<typename T>
    cxx::future<T> schedule(cxx::function<T()> const& task);
    void drain();
    bool empty();
    int64_t size();
    int64_t queueLength();

    //
    // Internal implementation
    //

    /// Execute an available task.
    ///
    /// Called by worker threads to execute the next available task,
    /// or to block until a task is available.
    ///
    /// @param[in]      worker      the calling worker
    ///
    /// @return         true if the worker should continue, false to exit
    bool runTask(Worker *worker);

    /// Signal that a woker has terminated unexpectedly (exception).
    ///
    /// @param[in]      worker      the worker
    ///
    void workerTerminatedUnexpectedly(Worker *worker);
private:
    void initPool();
    void shutdown();
    bool addThread();
    void workerTerminated(Worker *worker, bool expected);

    const int64_t minPoolSize_;
    const int64_t maxPoolSize_;
    const int64_t keepAlive_;

    int64_t poolSize_; ///< Existing workers in the pool
    int64_t activeWorkers_; ///< Workers running tasks
    bool shutdown_; ///< Whether shutdown has been triggered

    std::deque<cxx::function<void()>> tasks_; ///< Outstanding tasks
    std::vector<Worker *> terminated_; ///< Terminated workers to join

    typedef cxx::mutex Mutex;

    Mutex mutex_;
    cxx::condition_variable_any taskCv_;
    cxx::condition_variable_any drainCv_;
};

} // detail namespace
} // maginatics namespace

#include <maginatics/threadpool/detail/pool-impl.h>

#endif // MAGINATICS_THREADPOOL_DETAIL_POOL_H_
