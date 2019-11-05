#include "KnnAlgorithm.h"

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

float KnnAlgorithm::predict() {
    int closest_neighbour_index, accurate_predictions = 0;
    float max_float = std::numeric_limits<float>::max();

    for (int current_test_row=0; current_test_row < test_rows; ++current_test_row) {
        float closest_neighbour_distance = max_float;
        float* tst = test_data + (columns * current_test_row);
        // for each row in train dataset
        for (int i = 0; i < train_rows; ++i) {
            float* tr = train_data + (i * columns) + 1;
            // calculate eucidlean metric and get the closest one
            float sum = 0;
            for(int j = 1; j < columns; ++j, ++tr) {
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
            accurate_predictions = accurate_predictions + 1;
        }
    }

    //printf("Czas obliczen knn: %f\n", MPI_Wtime() - startTime);

    return (accurate_predictions / float(test_rows)) * 100;
}
