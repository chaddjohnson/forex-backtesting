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

#ifndef MAGINATICS_THREADPOOL_THREADPOOL_H_
#define MAGINATICS_THREADPOOL_THREADPOOL_H_

#include <maginatics/threadpool/detail/pool.h>
#include <maginatics/threadpool/detail/port.h>

namespace maginatics {

/// An elastic FIFO thread pool with reference semantics.
///
/// Example usage:
///
///     // Pool with at least 1 and no more than 512 workers;
///     // idle workers reaped after 30 seconds
///     ThreadPool pool(1, 512, 30000);
///     for (int i = 0; i < 1E6; ++i) {
///         pool.execute([]() { someTask(); });
///     }
///     pool.drain();
///
class ThreadPool {
public:
    /// Creates a new thread pool with provided parameters.
    ///
    /// @param[in]      minPoolSize     minimum number of threads
    /// @param[in]      maxPoolSize     maximum number of threads
    /// @param[in]      keepAlive       keep alive for threads > than the
    ///                                 minimum, in milliseconds
    ThreadPool(int64_t minPoolSize,
               int64_t maxPoolSize,
               int64_t keepAlive)
        : pool_(cxx::make_shared<detail::Pool>(
                    minPoolSize, maxPoolSize, keepAlive)) { }

    /// Createa a new fixed-size thread pool.
    ///
    /// @param[in]      poolSize        fixed pool size
    ///
    explicit ThreadPool(int64_t poolSize)
        : pool_(cxx::make_shared<detail::Pool>(
                    poolSize, poolSize, 0)) { }

    /// Execute a task at a later time.
    ///
    /// @param[in]      task        the task
    ///
    /// @return                     true, or false in case of a terrible error
    ///
    bool execute(cxx::function<void()> const& task) {
        return pool_->execute(task);
    }

    /// Execute a task that returns a value at a later time.
    ///
    /// @param[in]      task        the task
    ///
    /// @return                     the future result of the task
    ///
    template<typename T>
    cxx::future<T> schedule(cxx::function<T()> const& task) {
        return pool_->schedule(task);
    }

    /// Wait for all tasks to complete.
    ///
    /// Blocks until all scheduled tasks have executed.
    ///
    void drain() {
        pool_->drain();
    }

    /// @return         whether the pool is empty (has no tasks)
    ///
    bool empty() const {
        return pool_->empty();
    }

    /// @return         the number of executing tasks in the pool
    ///
    int64_t size() {
        return pool_->size();
    }

    /// @return         the number of pending tasks in the pool
    ///
    int64_t queueLength() {
        return pool_->queueLength();
    }

private:
    cxx::shared_ptr<detail::Pool> pool_; ///< Pool implementation
};

} // maginatics namespace

#endif // MAGINATICS_THREADPOOL_THREADPOOL_H_
