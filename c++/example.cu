#include <stdio.h>
static void HandleError( cudaError_t err, const char *file,  int line ) {
        if (err != cudaSuccess) {
                printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
                exit( EXIT_FAILURE );
        }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void divide_array_by_column( float *data, int row_size, int col_size) {
    int sectionsCount = blockDim.x * gridDim.x;
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("sid=%d\n", sid);
    for (int i = sid; i < row_size * col_size; i = i + sectionsCount) {
        printf("sid=%d value=%f\n", sid, *(data + i));
    }
}

__global__ void divide_array_by_row( float *data, int row_size, int col_size) {
    int sectionsCount = blockDim.x * gridDim.x;
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("sid=%d\n", sid);
    while (sid < col_size) {
        for (int i = 0; i < row_size; i++) {
            printf("sid=%d value=%f\n", sid, *(data + (sid * row_size) + i));
        }
        sid += sectionsCount;
    }

}

void print_device_info(int i) {
    cudaDeviceProp prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i) );
    printf( " --- Ogólne informacje o urządzeniu %d — \n", i );
    printf( "Nazwa: %s\n", prop.name );
    printf( "Potencjał obliczeniowy: %d.%d\n", prop.major, prop.minor );
    printf( "Zegar: %d\n", prop.clockRate );
    printf( "Ustawienie deviceOverlap: " );
    if (prop.deviceOverlap)
        printf( "Włączone\n" );
    else
        printf( "Wyłączone\n" );
    printf( "Limit czasu dziatania jądra: " );
    if (prop.kernelExecTimeoutEnabled)
        printf( "Wyłączony\n" );
    else
        printf( "Włączony\n" );
    printf( " — Informacje o pamięci urządzenia %d — '\n", i );
    printf( "Ilość pamięci globalnej: %ld\n", prop.totalGlobalMem );
    printf( "Ilość pamięci stałej: %ld\n", prop.totalConstMem );
    printf( "Maks. szerokość pamięci: %ld\n", prop.memPitch );
    printf( "Wyrównanie tekstur: %ld\n", prop.textureAlignment );
    printf( " — - Informacje na temat wieloprocesorów urządzenia %d \n", i );
    printf( "Liczba wieloprocesorów: %d\n",
            prop.multiProcessorCount );
    printf( "Pamięć wspólna na wieloprocesor: %ld\n", prop.sharedMemPerBlock );
    printf( "Rejestry na wieloprocesor: %d\n", prop.regsPerBlock );
    printf( "Liczba wątków w osnowie: %d\n", prop.warpSize );
    printf( "Maks. liczba wątków na blok: %d\n",
            prop.maxThreadsPerBlock );
    printf( "Maks. liczba wymiarów wątków: (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2] );
    printf( "Maks. liczba wymiarów siatki: (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2] );
    printf( "\n" );
}
	
int main(void) {
    print_device_info(0);
    float *cuda_data;
    int data_col_size = 5;
    int data_row_size = 4;
	float data_array[] = {1,  2,  3,  4,
                          5,  6,  7,  8,
                          9,  10, 11, 12,
                          13, 14, 15, 16,
                          17, 18, 19, 20};

	HANDLE_ERROR(cudaMalloc( (void**)&cuda_data, sizeof(float) * data_col_size * data_row_size));
	HANDLE_ERROR(cudaMemcpy( cuda_data, data_array, sizeof(float) * data_col_size * data_row_size ,
	        cudaMemcpyHostToDevice  ));

    divide_array_by_column<<<2,2>>>(cuda_data, data_row_size, data_col_size);
    //divide_array_by_row<<<2,2>>>(cuda_data, data_row_size, data_col_size);

    cudaFree(cuda_data );
	return 0;
}