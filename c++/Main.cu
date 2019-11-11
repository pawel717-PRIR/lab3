#include "LoadCSV.h"
#include "Data.h"
#include "Preprocessing.h"
#include "KnnAlgorithm.h"
#include <stdio.h>
#include <ctime>

using namespace std;
/**
 * compilation:
 * nvcc Main.cu Data.cu KnnAlgorithm.cu LoadCSV.cu Preprocessing.cu
 * @return
 */
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Wrong arguments. Proper invocation is <blocks_count> <threads_count_per_block>");
    }
    int blocks_count = atoi(argv[1]);
    int threads_count_per_block = atoi(argv[2]);

    Data ourData = Data();
    Preprocessing preprocessing = Preprocessing();
    KnnAlgorithm knn = KnnAlgorithm();

    LoadCSV().myLoad("../dataset/mnist_train.csv", ourData.data, ourData.rows, ourData.columns);
	preprocessing.Normalization(ourData.data, ourData.rows, ourData.columns,
	        threads_count_per_block, blocks_count);
//    preprocessing.Standarization(ourData.data, ourData.rows, ourData.columns,
//            threads_count_per_block, blocks_count);
    knn.fit(&ourData, 75);
    float accuracy = knn.predict(threads_count_per_block, blocks_count);
    printf("Accuracy knn: %f\n", accuracy);

	return 0;
}
