/* 
 * File:   kernel.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on 13 oktober 2015, 0:33
 */

#ifndef KERNEL_CUH
#define	KERNEL_CUH
__global__ void cudaKernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount);

__global__ void cudaSpinKernel(DataArray dataArray, SpinDataArray spinDataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, SpinMap spinMap, int particleCount, int iterationCount);

#endif	/* KERNEL_CUH */

