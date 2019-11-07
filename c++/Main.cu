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
int main() {

    Data ourData = Data();
    Preprocessing preprocessing = Preprocessing();
    KnnAlgorithm knn = KnnAlgorithm();

    LoadCSV().myLoad("../dataset/mnist_train.csv", ourData.data, ourData.rows, ourData.columns);
	//preprocessing.Normalization(ourData.data, ourData.rows, ourData.columns);
    preprocessing.Standarization(ourData.data, ourData.rows, ourData.columns);
    knn.fit(&ourData, 75);
    float accuracy = knn.predict();
    //printf("Accuracy knn: %f\n", accuracy);

	return 0;
}
