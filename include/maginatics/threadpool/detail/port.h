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

#ifndef MAGINATICS_THREADPOOL_DETAIL_PORT_H_
#define MAGINATICS_THREADPOOL_DETAIL_PORT_H_

#if defined(USE_BOOST_THREADING)

#if !defined(BOOST_THREAD_VERSION)
#define BOOST_THREAD_VERSION 4
#elif BOOST_THREAD_VERSION < 4
#pragma message ("Boost thread version 4+ required")
#endif

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/future.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

namespace maginatics {
namespace cxx {
    using boost::bind;
    using boost::condition_variable_any;
    using boost::function;
    using boost::future;
    using boost::make_shared;
    using boost::move;
    using boost::mutex;
    using boost::packaged_task;
    using boost::ref;
    using boost::shared_ptr;
    using boost::thread;
    using boost::unique_lock;

    namespace chrono { using boost::chrono::milliseconds; }
}}

#else
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <thread>

namespace maginatics {
namespace cxx {
    using std::bind;
    using std::condition_variable_any;
    using std::function;
    using std::future;
    using std::make_shared;
    using std::move;
    using std::mutex;
    using std::packaged_task;
    using std::ref;
    using std::shared_ptr;
    using std::thread;
    using std::unique_lock;

    namespace chrono { using std::chrono::milliseconds; }
}}

#endif

#endif // MAGINATICS_THREADPOOL_DETAIL_PORT_H_
