#include "KnnAlgorithm.h"
static void HandleError( cudaError_t err, const char *file,  int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
__global__ void cuda_knn_predict(float *data, int train_rows, int test_rows, int columns, int * accurate_predictions) {
    int total_threads_count = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int closest_neighbour_index;
    float max_float = FLT_MAX;
    float* train_data = data;
    float* test_data = data + (columns * train_rows);

    for (int current_test_row=tid; current_test_row < test_rows; current_test_row=current_test_row+total_threads_count) {
        float closest_neighbour_distance = max_float;
        float* tst = test_data + (columns * current_test_row);
        // for each row in train dataset
        for (int i = 0; i < train_rows; ++i) {
            float* tr = train_data + (i * columns) + 1;
            // calculate eucidlean metric and get the closest one
            float sum = 0;
            for (int j = 1; j < columns; ++j, ++tr) {
                float difference = *(tr) - *(tst +j);
                sum = sum + (difference * difference);
            }
            // distance is euclidean metric for current_test_row and i-th train data
            // if our data is closer to that row from train data update closest_neighbour_distance and and closest_neighbour_index
            if(sum < closest_neighbour_distance) {
                closest_neighbour_distance = sum;
                closest_neighbour_index = i;
            }
        }
        // now we have found closest neighbour and have index of that neighbour in closest_neighbour_index variable
        // so let's get target class of that neighbour (predicted class) and check if the prediction is accurate
        if(*(test_data + (current_test_row * columns)) == *(train_data + (closest_neighbour_index * columns))) {
            // if prediction is accurate increment accurate predictions counter
            atomicAdd(accurate_predictions, 1);
        }
    }
}

KnnAlgorithm::KnnAlgorithm() {
    //ctor
}

KnnAlgorithm::~KnnAlgorithm() {
    //dtor
}

void KnnAlgorithm::fit(Data * data, int percent) {
    this->train_rows = (data->rows * percent) / 100;
    this->columns = data->columns;
    this->test_rows = data->rows - train_rows;
    this->train_data = data->data;
    this->test_data = data->data + (columns * train_rows);
}

float KnnAlgorithm::predict(int threads_count_per_block, int blocks_count ) {
    int accurate_predictions;
    int * cuda_accurate_predictions;

    cudaDeviceProp cuda_properties; // information about gpu
    HANDLE_ERROR(cudaGetDeviceProperties( &cuda_properties, 0));

    // copy data to compute into gpu device memory
    float *cuda_data;
    int data_size = sizeof(float) * (this->test_rows + this->train_rows) * this->columns;
    HANDLE_ERROR(cudaMalloc((void**)&cuda_data, data_size));
    HANDLE_ERROR(cudaMemcpy(cuda_data, this->train_data, data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&cuda_accurate_predictions,  sizeof(int)));

    // measure time using cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // perform knn prediction
    cuda_knn_predict<<<blocks_count, threads_count_per_block>>>(cuda_data, this->train_rows, this->test_rows, this->columns, cuda_accurate_predictions);
    cudaEventRecord(stop);

    // print elapsed time
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Czas obliczen knn: %f\n", elapsed_time/1000);

    // copy from gpu device memory to host RAM
    HANDLE_ERROR(cudaMemcpy(&accurate_predictions, cuda_accurate_predictions, sizeof(int),
            cudaMemcpyDeviceToHost));

    return (accurate_predictions / float(test_rows)) * 100;
}
