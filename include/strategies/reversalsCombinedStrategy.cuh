#ifndef REVERSALSCOMBINEDSTRATEGY_H
#define REVERSALSCOMBINEDSTRATEGY_H

#include "combinedStrategy.cuh"
#include "types/real.cuh"

class ReversalsCombinedStrategy : public Strategy {
    private:


    protected:


    public:
        __device__ __host__ ReversalsCombinedStrategy();
        __device__ __host__ ~ReversalsCombinedStrategy() {}
};

#endif
