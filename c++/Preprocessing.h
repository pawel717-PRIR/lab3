#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <float.h>

class Preprocessing
{
public:
	Preprocessing();
	~Preprocessing();
    void Normalization(float *data, int rows, int columns);
	void Standarization(float *data, int rows, int columns);
};

