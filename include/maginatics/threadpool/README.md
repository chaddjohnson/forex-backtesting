Maginatics ThreadPool
=====================

ThreadPool is a C++ implementation of the thread pool pattern, allowing
arbitrary tasks to be queued for execution by a dedicated pool of worker
threads. It is a header-only library and requires
[Boost](http://http://www.boost.org/) with support for version 4 of the
`Boost.Thread` interface. Nathan Rosenblum at Maginatics
<nater@maginatics.com> originally wrote ThreadPool.

Overview
--------

ThreadPool implements a bounded thread pool with a minimum core pool size
and configurable keep-alive for idle threads; those familiar with the Java
[ThreadPoolExecutor](http://docs.oracle.com/javase/7/docs/api/java/util/concurrent/ThreadPoolExecutor.html)
will find the interface and semantics familiar. At its core is a pool of
preallocated worker threads. New tasks are executed immediately if idle
workers are available or if new workers can be allocated without exceeding the
limit; otherwise, the tasks are added to a FIFO queue for later execution. A
fixed core pool of one or more threads is maintained, with idle threads above
this limit terminating after a configurable keep-alive period.

Example use
-----------

The `ThreadPool` constructor takes `minPoolSize`, `maxPoolSize`, and
`keepAlive` parameters; the following example demonstrates scheduling a large
number of tasks for execution on a pool with a maximum thread count of 512 and
a 30 second keep-alive timeout:

    void someTask();

    ThreadPool pool(1, 512, 30000);
    for (int i = 0; i < 1E6; ++i) {
        pool.execute([]() { someTask(); });
    }
    pool.drain();

Task return values can be retrieved using the `schedule` interface:

    bool someNonVoidTask();

    ThreadPool pool(1, 512, 30000);
    auto result = pool.schedule<bool>([]() -> bool {
                return someNonVoidTask();
            });
    printf("Returned %d\n", result.get());

Refer to the documentation for further details.

Generating documentation
------------------------

Details of the ThreadPool interface are provided in the form of
[Doxygen](http://www.stack.nl/~dimitri/doxygen/)-formatted comments in the
code. To produce an HTML version of the documentation, execute

    ./waf configure build

in the project root. This will produce documentation in the
`${PROJECT_ROOT}/build/doc` directory.

License
-------
Copyright (C) 2012-2013 Maginatics, Inc.

Licensed under the MIT License. The [waf](https://code.google.com/p/waf/)
build system distributed alongside this software is licensed under the BSD
3-Clause License.