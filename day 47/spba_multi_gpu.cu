#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>
#include <chrono>

#define CHECK_HIP_ERROR(err) do { \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__ void evaluateCandidatesKernel(const float* candidates, float* fitness, int numCandidates, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCandidates) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float x = candidates[idx * d + j];
            sum += x * x;
        }
        fitness[idx] = sum;
    }
}

struct Candidate {
    std::vector<float> position;
    float fitness;
};

void evaluateCandidatesOnDevice(int device, const float* candidates, float* fitness, int numCandidates, int d) {
    hipError_t err;
    err = hipSetDevice(device);
    CHECK_HIP_ERROR(err);

    size_t dataSize = numCandidates * d * sizeof(float);
    float* d_candidates = nullptr;
    float* d_fitness = nullptr;
    err = hipMalloc(&d_candidates, dataSize);
    CHECK_HIP_ERROR(err);
    err = hipMalloc(&d_fitness, numCandidates * sizeof(float));
    CHECK_HIP_ERROR(err);

    err = hipMemcpy(d_candidates, candidates, dataSize, hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(err);

    int threadsPerBlock = 256;
    int blocks = (numCandidates + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(evaluateCandidatesKernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_candidates, d_fitness, numCandidates, d);
    err = hipDeviceSynchronize();
    CHECK_HIP_ERROR(err);

    err = hipMemcpy(fitness, d_fitness, numCandidates * sizeof(float), hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(err);

    hipFree(d_candidates);
    hipFree(d_fitness);
}

void evaluatePopulationMultiGPU(const std::vector<Candidate>& population, int d, int deviceCount, std::vector<float>& fitnessResults) {
    int total = population.size();
    std::vector<float> candidateData(total * d);
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < d; j++) {
            candidateData[i * d + j] = population[i].position[j];
        }
    }
    fitnessResults.resize(total);

    int partitionSize = (total + deviceCount - 1) / deviceCount;
    std::vector<std::thread> threads;
    for (int dev = 0; dev < deviceCount; dev++) {
        int start = dev * partitionSize;
        int end = std::min(start + partitionSize, total);
        if (start >= end) break;
        int numCandidates = end - start;
        float* candidatesSubset = candidateData.data() + start * d;
        float* fitnessSubset = fitnessResults.data() + start;
        threads.emplace_back(evaluateCandidatesOnDevice, dev, candidatesSubset, fitnessSubset, numCandidates, d);
    }
    for (auto& t : threads) {
        t.join();
    }
}

Candidate randomCandidate(int d, float range, std::mt19937& rng) {
    Candidate cand;
    cand.position.resize(d);
    std::uniform_real_distribution<float> dist(-range, range);
    for (int i = 0; i < d; i++) {
        cand.position[i] = dist(rng);
    }
    cand.fitness = 0.0f;
    return cand;
}

int main() {
    const int populationSize = 256;
    const int dimension = 30;
    const int maxIter = 100;
    const float alpha = 0.1f;
    const int numBestSites = populationSize / 10;
    const int recruitsPerSite = 10;
    const float searchRange = 10.0f;

    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<Candidate> population;
    for (int i = 0; i < populationSize; i++) {
        population.push_back(randomCandidate(dimension, searchRange, rng));
    }

    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount < 1) {
        std::cerr << "No HIP devices found!" << std::endl;
        return -1;
    }
    std::cout << "Found " << deviceCount << " HIP device(s).\n";

    for (int iter = 0; iter < maxIter; iter++) {
        std::vector<float> fitnessResults;
        evaluatePopulationMultiGPU(population, dimension, deviceCount, fitnessResults);
        for (int i = 0; i < populationSize; i++) {
            population[i].fitness = fitnessResults[i];
        }

        std::sort(population.begin(), population.end(), [](const Candidate& a, const Candidate& b) {
            return a.fitness < b.fitness;
        });

        std::cout << "Iteration " << iter << ", Best fitness: " << population[0].fitness << std::endl;

        for (int i = 0; i < numBestSites; i++) {
            Candidate bestCandidate = population[i];
            for (int r = 0; r < recruitsPerSite; r++) {
                Candidate newCandidate = bestCandidate;
                for (int j = 0; j < dimension; j++) {
                    std::uniform_real_distribution<float> perturbDist(-alpha, alpha);
                    newCandidate.position[j] += perturbDist(rng);
                }
                float fitness = 0.0f;
                for (int j = 0; j < dimension; j++) {
                    fitness += newCandidate.position[j] * newCandidate.position[j];
                }
                newCandidate.fitness = fitness;
                if (newCandidate.fitness < bestCandidate.fitness) {
                    bestCandidate = newCandidate;
                }
            }
            population[i] = bestCandidate;
        }

        for (int i = populationSize / 2; i < populationSize; i++) {
            population[i] = randomCandidate(dimension, searchRange, rng);
        }
    }

    std::cout << "Optimization completed. Best solution fitness: " << population[0].fitness << std::endl;
    return 0;
}
