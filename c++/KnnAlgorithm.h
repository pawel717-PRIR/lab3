#pragma once
#include "Data.h"
#include <stdio.h>
#include <float.h>

class KnnAlgorithm {
    public:
        KnnAlgorithm();
        virtual ~KnnAlgorithm();
        void fit(Data * data, int percent);
        float predict();
    private:
        float *train_data;
        float *test_data;
        int train_rows;
        int test_rows;
        int columns;
};
