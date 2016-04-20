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

#ifndef MAGINATICS_THREADPOOL_DETAIL_POOL_IMPL_H_
#define MAGINATICS_THREADPOOL_DETAIL_POOL_IMPL_H_

#include <cassert>

#include <maginatics/threadpool/detail/port.h>
#include <maginatics/threadpool/detail/pool.h>
#include <maginatics/threadpool/detail/worker.h>

namespace maginatics {
namespace detail {

inline void Pool::initPool() {
    cxx::unique_lock<Mutex> lock(mutex_);
    // Bring up the minimum number of workers
    for (int i = 0; i < minPoolSize_; ++i) {
        addThread();
    }
}

inline bool Pool::empty() {
    cxx::unique_lock<Mutex> lock(mutex_);
    return tasks_.empty();
}

inline int64_t Pool::size() {
    cxx::unique_lock<Mutex> lock(mutex_);
    return poolSize_;
}

inline int64_t Pool::queueLength() {
    cxx::unique_lock<Mutex> lock(mutex_);
    return tasks_.size();
}

inline void Pool::drain() {
    cxx::unique_lock<Mutex> lock(mutex_);
    while (activeWorkers_ != 0 || !tasks_.empty()) {
        drainCv_.wait(lock);
    }
}

inline void Pool::shutdown() {
    cxx::unique_lock<Mutex> lock(mutex_);
    shutdown_ = true;
    taskCv_.notify_all();
    while (poolSize_ > 0) {
        drainCv_.wait(lock);
    }
    // Reap dead workers
    for (auto worker : terminated_) {
        worker->join();
        delete worker;
    }
    terminated_.clear();
}

inline bool Pool::execute(cxx::function<void()> const& task) {
    cxx::unique_lock<Mutex> lock(mutex_);

    if (poolSize_ < minPoolSize_) {
        // Restore any missing worker threads
        addThread();
    } else if (poolSize_ < maxPoolSize_ &&
            (static_cast<int64_t>(tasks_.size()) + activeWorkers_
                >= poolSize_)) {
        // Add another worker if we can't immediately schedule
        addThread();
    }

    // This should never happen
    if (poolSize_ == 0) {
        return false;
    }

    tasks_.push_back(task);
    taskCv_.notify_one();

    return true;
}

template<typename T>
cxx::future<T> Pool::schedule(cxx::function<T()> const& task) {
    // Looking *so* forward to std::bind, which understands move semantics
    // even if lambdas do not. In the meantime, create a packaged task
    // pointer wrapper so that we can pass something into our lambda.
    // Le sigh.
    typedef cxx::shared_ptr<cxx::packaged_task<T()>> Wrapper;

    // TODO(nater): The static cast converts task into an rvalue and forces
    // a copy working around Boost #8596, which is fixed in 1.54. Though
    // the aforementioned move semantics would also fix it :/
    Wrapper packaged(cxx::make_shared<cxx::packaged_task<T()>> (
            static_cast<cxx::function<T()>> (task)));
    cxx::future<T> ret(packaged->get_future());

    execute(cxx::bind<void>(
        [](Wrapper const& wrapper) {
            (*wrapper)();
        }, packaged));

    // MSVC needs a hint
    return cxx::move(ret);
}

inline bool Pool::addThread() {
    Worker *worker = new Worker(this);
    if (worker->start()) {
        ++poolSize_;
        ++activeWorkers_;
        return true;
    }
    delete worker;
    return false;
}

inline void Pool::workerTerminatedUnexpectedly(Worker *worker) {
    cxx::unique_lock<Mutex> lock(mutex_);
    workerTerminated(worker, false);
}

// Lock must be held
inline void Pool::workerTerminated(Worker *worker, bool expected) {
    --poolSize_;
    --activeWorkers_;

    assert(poolSize_ >= 0);
    assert(activeWorkers_ >= 0);

    if (!expected) {
        // Add a new thread if necessary
        if (!shutdown_ && poolSize_ < minPoolSize_) {
            addThread();
        }
    }

    terminated_.push_back(worker);
    drainCv_.notify_all();
}

inline bool Pool::runTask(Worker *worker) {
    cxx::function<void()> task;

    cxx::unique_lock<Mutex> lock(mutex_);
    if (!terminated_.empty()) {
        // Reap dead workers
        for (auto worker : terminated_) {
            worker->join();
            delete worker;
        }
        terminated_.clear();
    }

    while (tasks_.empty() && !shutdown_) {
        --activeWorkers_;
        // Kick drain / exit
        drainCv_.notify_all();
        if (poolSize_ > minPoolSize_) {
            // Wait up to the keepAlive for a task to show up
            taskCv_.wait_for(lock, cxx::chrono::milliseconds(keepAlive_));
            if (poolSize_ > minPoolSize_ && tasks_.empty()) {
                ++activeWorkers_; // For accounting
                workerTerminated(worker, true); // Expected termination
                return false; // Break out of run loop
            }
        } else {
            taskCv_.wait(lock);
        }
        ++activeWorkers_;
    }

    if (!shutdown_) {
        task = tasks_.front();
        tasks_.pop_front();
    } else {
        workerTerminated(worker, true); // Expected termination
        return false;
    }

    lock.unlock();

    // Consumers may have queued empty tasks
    if (task) {
        task();
    }

    return true;
}

} // detail namespace
} // maginatics namespace

#endif // MAGINATICS_THREADPOOL_DETAIL_POOL_IMPL_H_
