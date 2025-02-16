#pragma once

class SGEMM {
public:
    virtual void init() = 0;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) = 0;
    virtual void finalize() = 0;
    virtual ~SGEMM() = default;
}; 