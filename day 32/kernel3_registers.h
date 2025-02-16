#pragma once
#include "sgemm.h"

class Kernel3Registers : public SGEMM {
public:
    void init() override;
    void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    void finalize() override;
}; 