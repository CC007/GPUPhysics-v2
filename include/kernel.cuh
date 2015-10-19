/* 
 * File:   kernel.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on 13 oktober 2015, 0:33
 */

#ifndef KERNEL_CUH
#define	KERNEL_CUH

__device__ void cudaSumArrayHelper(double *nums, int length, int interval);

__device__ double cudaSumArray(double *nums, int length);

__device__ void cudaCalcData(DataArray dataArray, int iteration, Map map, double *newValue);

__global__ void cudaKernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount);

#endif	/* KERNEL_CUH */

