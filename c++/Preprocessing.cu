#include "Preprocessing.h"

Preprocessing::Preprocessing() {
}


Preprocessing::~Preprocessing() {
}

void Preprocessing::Normalization(float *data, int rows, int columns) {
    int min, max, col_start_index = 0;
    float max_float = std::numeric_limits<float>::max();

    for (int i = col_start_index; i < columns; ++i) {
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

   // printf("Czas obliczen normalizacja: %f\n", MPI_Wtime() - startTime);
}


void Preprocessing::Standarization(float *data, int rows, int columns) {
    int col_start_index = 0;
    float var, ave, amo;

    for (int i = col_start_index; i < columns - 1; ++i) {
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

    //printf("Czas obliczen standaryzacja: %f\n", MPI_Wtime() - startTime);
}