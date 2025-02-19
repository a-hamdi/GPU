#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define G 6.67430e-11  // Gravitational constant
#define DT 1e-3        // Time step
#define STEPS 1000     // Simulation steps

struct Body {
    float3 position;
    float3 velocity;
    float mass;
};

__device__ float3 computeAcceleration(Body* bodies, int id) {
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 3; i++) {
        if (i != id) {
            float3 r;
            r.x = bodies[i].position.x - bodies[id].position.x;
            r.y = bodies[i].position.y - bodies[id].position.y;
            r.z = bodies[i].position.z - bodies[id].position.z;

            float distSq = r.x * r.x + r.y * r.y + r.z * r.z + 1e-9;  // Avoid division by zero
            float dist = sqrtf(distSq);
            float force = G * bodies[i].mass / distSq;

            acc.x += force * r.x / dist;
            acc.y += force * r.y / dist;
            acc.z += force * r.z / dist;
        }
    }
    return acc;
}

__global__ void updateBodies(Body* d_bodies) {
    int id = threadIdx.x;

    if (id < 3) {
        float3 acc = computeAcceleration(d_bodies, id);

        d_bodies[id].velocity.x += acc.x * DT;
        d_bodies[id].velocity.y += acc.y * DT;
        d_bodies[id].velocity.z += acc.z * DT;

        d_bodies[id].position.x += d_bodies[id].velocity.x * DT;
        d_bodies[id].position.y += d_bodies[id].velocity.y * DT;
        d_bodies[id].position.z += d_bodies[id].velocity.z * DT;
    }
}

void simulate() {
    Body h_bodies[3] = {
        {{-1.0f, 0.0f, 0.0f}, {0.0f, -0.5f, 0.0f}, 1.0e10f},
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.5f, 0.0f}, 1.0e10f},
        {{0.0f, 1.0f, 0.0f}, {0.5f, 0.0f, 0.0f}, 1.0e10f}
    };

    Body* d_bodies;
    cudaMalloc((void**)&d_bodies, sizeof(h_bodies));
    cudaMemcpy(d_bodies, h_bodies, sizeof(h_bodies), cudaMemcpyHostToDevice);

    for (int i = 0; i < STEPS; i++) {
        updateBodies<<<1, 3>>>(d_bodies);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_bodies, d_bodies, sizeof(h_bodies), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);

    std::cout << "Final Positions:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "Body " << i << ": (" << h_bodies[i].position.x << ", " 
                  << h_bodies[i].position.y << ", " << h_bodies[i].position.z << ")\n";
    }
}

int main() {
    simulate();
    return 0;
}
