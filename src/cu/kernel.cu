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
	nums = (double*) malloc(map->length * sizeof (double));
	memset(nums, 0, map->length * sizeof (double));
	/*if (safeDeviceCalloc((void**) &nums, map->length, sizeof (double))) {
		printf("Unable to to calculate iteration %d (local variable for the calculation could not be allocated", iteration);
	}*/

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
	free(nums);
	//safeDeviceFree((void**) &nums);
}

__global__ void cudaKernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount) {
	int sizeX = gridDim.x;
	int iteration = blockIdx.x;
	int sizeY = blockDim.x;
	int idy = threadIdx.x;
	for (int n = iteration * sizeY + idy; n < particleCount; n += sizeX * sizeY) {
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