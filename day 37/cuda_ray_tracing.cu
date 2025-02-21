#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 768
#define SPHERE_RADIUS 0.5f
#define SPHERE_CENTER_X 0.0f
#define SPHERE_CENTER_Y 0.0f
#define SPHERE_CENTER_Z -1.5f

struct Vec3 {
    float x, y, z;
    
    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    __device__ Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __device__ float dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    __device__ Vec3 normalize() const {
        float len = sqrtf(x*x + y*y + z*z);
        return Vec3(x/len, y/len, z/len);
    }
};

// CUDA kernel to trace rays
__global__ void render(unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = (y * WIDTH + x) * 3; // Each pixel has 3 color channels (RGB)

    // Convert pixel to normalized screen space (-1 to 1)
    float u = (2.0f * x / WIDTH - 1.0f);
    float v = (2.0f * y / HEIGHT - 1.0f);

    // Ray origin (camera at (0,0,0))
    Vec3 ray_origin(0.0f, 0.0f, 0.0f);
    // Ray direction (from camera to screen)
    Vec3 ray_dir(u, v, -1.0f);
    ray_dir = ray_dir.normalize();

    // Sphere properties
    Vec3 sphere_center(SPHERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_CENTER_Z);
    
    // Compute ray-sphere intersection (quadratic equation)
    Vec3 oc = ray_origin - sphere_center;
    float a = ray_dir.dot(ray_dir);
    float b = 2.0f * oc.dot(ray_dir);
    float c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
        // Compute the nearest intersection
        float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        Vec3 hit_point = ray_origin + ray_dir * t;
        Vec3 normal = (hit_point - sphere_center).normalize();

        // Simple diffuse shading based on light direction
        Vec3 light_dir(1.0f, 1.0f, -1.0f);
        light_dir = light_dir.normalize();
        float intensity = fmaxf(0.0f, normal.dot(light_dir));

        // Color the sphere (red tone)
        image[idx] = (unsigned char)(255 * intensity);
        image[idx + 1] = (unsigned char)(50 * intensity);
        image[idx + 2] = (unsigned char)(50 * intensity);
    } else {
        // Background color (black)
        image[idx] = 0;
        image[idx + 1] = 0;
        image[idx + 2] = 0;
    }
}

// Save image as PPM
void save_image(unsigned char *image) {
    FILE *f = fopen("output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, 1, WIDTH * HEIGHT * 3, f);
    fclose(f);
}

int main() {
    // Allocate memory for image
    unsigned char *d_image, *h_image;
    size_t image_size = WIDTH * HEIGHT * 3;
    cudaMalloc((void **)&d_image, image_size);
    h_image = (unsigned char *)malloc(image_size);

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    render<<<numBlocks, threadsPerBlock>>>(d_image);
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    // Save image
    save_image(h_image);

    // Cleanup
    cudaFree(d_image);
    free(h_image);

    printf("Image saved as output.ppm\n");
    return 0;
}
