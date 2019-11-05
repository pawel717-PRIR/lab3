#include "Preprocessing.h"
#include <float.h>
static void HandleError( cudaError_t err, const char *file,  int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void cuda_normalization(float *data, int rows, int columns) {
    int total_threads_count = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int min, max;
    float max_float = FLT_MAX;

    for (int i = tid+1; i < columns; i=i+total_threads_count) {
        min = max_float; max = 0;
        for (int j = 0; j < rows; ++j) {
            if (*(data + (j*columns)+i) < min) {
                min = *(data + (j*columns)+i);
            } else if (*(data + (j*columns)+i) > max) {
                max = *(data + (j*columns)+i);
            }
        }

        float max_min_reciprocal = max - min;
        if (max_min_reciprocal == 0) {
            continue;
        }
        max_min_reciprocal = 1. / max_min_reciprocal;

        for (int j = 0; j < rows; ++j) {
            *(data + (j*columns)+i) = (*(data + (j*columns)+i) - min) * max_min_reciprocal;
        }
    }
}

__global__ void cuda_standarization(float *data, int rows, int columns) {
    int total_threads_count = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float var, ave, amo;

    for (int i = tid+1; i < columns; i=i+total_threads_count) {
        amo = 0, var = 0;
        for (int j = 0; j < rows; ++j) {
            amo = amo + *(data + (j * columns) + i);
        }
        ave  = amo / float(rows);

        for (int j = 0; j < rows; ++j) {
            float factor = *(data + (j * columns) + i) - ave;
            var = var + (factor * factor);
        }

        if (var == 0) {
            for (int j = 0; j < rows; j++) {
                *(data + (j * columns) + i) = *(data + (j * columns) + i) / 255.;
            }
            continue;
        }

        float sd_reciprocal = 1./sqrt(var);

        for (int j = 0; j < rows; j++) {
            *(data + (j * columns) + i) = (*(data + (j * columns) + i) - ave) * sd_reciprocal;
        }
    }
}

Preprocessing::Preprocessing() {
}


Preprocessing::~Preprocessing() {
}

void Preprocessing::Normalization(float *data, int rows, int columns) {
    cudaDeviceProp cuda_properties; // information about gpu
    HANDLE_ERROR(cudaGetDeviceProperties( &cuda_properties, 0));
    int threads_count_per_block = cuda_properties.maxThreadsPerBlock; // use as many threads as possible on this device
    if(threads_count_per_block > columns) {
        threads_count_per_block = columns;
    }
    int blocks_count = (columns + threads_count_per_block - 1) / threads_count_per_block;
    int max_blocks_count = cuda_properties.maxGridSize[0];
    if(blocks_count > max_blocks_count) {
        blocks_count = max_blocks_count;
    }

    // copy data to compute into gpu device memory
    float *cuda_data;
    int data_size = sizeof(float) * rows * columns;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_data, data_size));
    HANDLE_ERROR(cudaMemcpy(cuda_data, data, data_size, cudaMemcpyHostToDevice));

    // measure time using cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // normalize
    cuda_normalization<<<blocks_count, threads_count_per_block>>>(cuda_data, rows, columns);
    cudaEventRecord(stop);

    // copy computed data to from gpu device memory to host RAM
    HANDLE_ERROR(cudaMemcpy(data, cuda_data, data_size, cudaMemcpyDeviceToHost));

    // print elapsed time
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Czas obliczen normalizacja: %f\n", elapsed_time/1000);
}


void Preprocessing::Standarization(float *data, int rows, int columns) {
    cudaDeviceProp cuda_properties; // information about gpu
    HANDLE_ERROR(cudaGetDeviceProperties( &cuda_properties, 0));
    int threads_count_per_block = cuda_properties.maxThreadsPerBlock; // use as many threads as possible on this device
    if(threads_count_per_block > columns) {
        threads_count_per_block = columns;
    }
    int blocks_count = (columns + threads_count_per_block - 1) / threads_count_per_block;
    int max_blocks_count = cuda_properties.maxGridSize[0];
    if(blocks_count > max_blocks_count) {
        blocks_count = max_blocks_count;
    }

    // copy data to compute into gpu device memory
    float *cuda_data;
    int data_size = sizeof(float) * rows * columns;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_data, data_size));
    HANDLE_ERROR(cudaMemcpy(cuda_data, data, data_size, cudaMemcpyHostToDevice));

    // measure time using cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // normalize
    cuda_standarization<<<blocks_count, threads_count_per_block>>>(cuda_data, rows, columns);
    cudaEventRecord(stop);

    // copy computed data to from gpu device memory to host RAM
    HANDLE_ERROR(cudaMemcpy(data, cuda_data, data_size, cudaMemcpyDeviceToHost));

    // print elapsed time
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Czas obliczen standaryzacja: %f\n", elapsed_time/1000);
}