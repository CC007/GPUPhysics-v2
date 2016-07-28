/* 
 * File:   kernel.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on Oktober 4, 2015, 4:38 PM
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../../include/safemem.cuh"
#include "../../include/data.h"
#include "../../include/map.h"
#include "../../include/spinmap.h"

__device__ void cudaSumArrayHelper(double *nums, int length, int interval) {
	int index = 0;
	int next = interval / 2;
	do {
		if (next < length) {
			nums[index] += nums[next];
		}
		index += interval;
		next += interval;
	} while (index < length);
}

__device__ double cudaSumArray(double *nums, int length) {
	if (length <= 0) {
		return 0;
	}
	int interval = 2;
	while (interval < length * 2) {
		cudaSumArrayHelper(nums, length, interval);
		interval *= 2;
	}
	return nums[0];
}

__device__ void cudaCalcData(DataArray dataArray, int iteration, Map map, double *newValue) {
	double *nums;
	//nums = (double*) malloc(map->length * sizeof (double));
	//memset(nums, 0, map->length * sizeof (double));
	if (map->length > 0) {
		if (safeDeviceCalloc((void**) &nums, map->length, sizeof (double))) {
			printf("Unable to to calculate iteration %d (local variable for the calculation could not be allocated).\n", iteration);
		}

		for (int i = 0; i < map->length; i++) {
			nums[i] = map->A[i]
					* pow(dataArray->x[iteration], (double) map->x[i])
					* pow(dataArray->dx[iteration], (double) map->dx[i])
					* pow(dataArray->y[iteration], (double) map->y[i])
					* pow(dataArray->dy[iteration], (double) map->dy[i])
					* pow(dataArray->delta[iteration], (double) map->delta[i])
					* pow(dataArray->phi[iteration], (double) map->phi[i]);
		}

		*newValue = cudaSumArray(nums, map->length);
		//free(nums);
		if (safeDeviceFree((void**) &nums)) {
			printf("Unable to free memory for local variable during calculation of iteration %d\n", iteration);
		}
	} else {
		*newValue = 0.0;
	}
}

__device__ void cudaCalcSpinRow(DataArray dataArray, int iteration, InnerSpinMap innerSpinMap, double *matrixRow) {
	cudaCalcData(dataArray, iteration, innerSpinMap->x, &(matrixRow[0]));
	cudaCalcData(dataArray, iteration, innerSpinMap->y, &(matrixRow[1]));
	cudaCalcData(dataArray, iteration, innerSpinMap->z, &(matrixRow[2]));
}

__device__ void cudaCalcSpin(DataArray dataArray, SpinDataArray spinDataArray, int iteration, SpinMap spinMap) {
	double **matrix;
	if (safeDeviceCalloc((void**) &matrix, 3, sizeof (double*))) {
		printf("Unable to create matrix for spin calculation (outer array)\n", iteration);
	}
	int i;
	for (i = 0; i < 3; i++) {
		if (safeDeviceCalloc((void**) &(matrix[i]), 3, sizeof (double*))) {
			printf("Unable to create matrix for spin calculation (inner array)\n", iteration);
		}
	}
	cudaCalcSpinRow(dataArray, iteration, spinMap->x, matrix[0]);
	cudaCalcSpinRow(dataArray, iteration, spinMap->y, matrix[1]);
	cudaCalcSpinRow(dataArray, iteration, spinMap->z, matrix[2]);

	spinDataArray->sx[iteration + 1] = matrix[0][0] * spinDataArray->sx[iteration] + matrix[0][1] * spinDataArray->sy[iteration] + matrix[0][2] * spinDataArray->sz[iteration];
	spinDataArray->sy[iteration + 1] = matrix[1][0] * spinDataArray->sx[iteration] + matrix[1][1] * spinDataArray->sy[iteration] + matrix[1][2] * spinDataArray->sz[iteration];
	spinDataArray->sz[iteration + 1] = matrix[2][0] * spinDataArray->sx[iteration] + matrix[2][1] * spinDataArray->sy[iteration] + matrix[2][2] * spinDataArray->sz[iteration];

	double divider = sqrt(spinDataArray->sx[iteration+1]^2 + spinDataArray->sy[iteration+1]^2 + spinDataArray->sz[iteration+1]^2)
	spinDataArray->sx[iteration+1] /= divider;
	spinDataArray->sy[iteration+1] /= divider;
	spinDataArray->sz[iteration+1] /= divider;

	for (i = 0; i < 3; i++) {
		if (safeDeviceFree((void**) &(matrix[i]))) {
			printf("Unable to free the matrix (inner array)\n");
		}

	}
	if (safeDeviceFree((void**) &matrix)) {
		printf("Unable to free the matrix (outer array)\n");
	}
}

__global__ void cudaKernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount) {
	int sizeX = gridDim.x;
	int itX = blockIdx.x;
	int sizeY = blockDim.x;
	int itY = threadIdx.x;
	for (int n = itX * sizeY + itY; n < particleCount; n += sizeX * sizeY) {
		for (int i = 0; i < iterationCount - 1; i++) {
			cudaCalcData(&(dataArray[n]), i, x, &(dataArray[n].x[i + 1]));
			cudaCalcData(&(dataArray[n]), i, dx, &(dataArray[n].dx[i + 1]));
			cudaCalcData(&(dataArray[n]), i, y, &(dataArray[n].y[i + 1]));
			cudaCalcData(&(dataArray[n]), i, dy, &(dataArray[n].dy[i + 1]));
			cudaCalcData(&(dataArray[n]), i, delta, &(dataArray[n].delta[i + 1]));
			cudaCalcData(&(dataArray[n]), i, phi, &(dataArray[n].phi[i + 1]));
		}
	}
}

__global__ void cudaSpinKernel(DataArray dataArray, SpinDataArray spinDataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, SpinMap spinMap, int particleCount, int iterationCount) {
	int sizeX = gridDim.x;
	int itX = blockIdx.x;
	int sizeY = blockDim.x;
	int itY = threadIdx.x;
	for (int n = itX * sizeY + itY; n < particleCount; n += sizeX * sizeY) {
		for (int i = 0; i < iterationCount - 1; i++) {
			cudaCalcData(&(dataArray[n]), i, x, &(dataArray[n].x[i + 1]));
			cudaCalcData(&(dataArray[n]), i, dx, &(dataArray[n].dx[i + 1]));
			cudaCalcData(&(dataArray[n]), i, y, &(dataArray[n].y[i + 1]));
			cudaCalcData(&(dataArray[n]), i, dy, &(dataArray[n].dy[i + 1]));
			cudaCalcData(&(dataArray[n]), i, delta, &(dataArray[n].delta[i + 1]));
			cudaCalcData(&(dataArray[n]), i, phi, &(dataArray[n].phi[i + 1]));
			cudaCalcSpin(&(dataArray[n]), &(spinDataArray[n]), i, spinMap);
		}
	}
}