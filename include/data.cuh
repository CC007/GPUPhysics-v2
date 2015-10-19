/* 
 * File:   data.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on October 13, 2015, 6:09 AM
 */

#ifndef DATA_CUH
#define	DATA_CUH

#include <cuda.h>
#include <cuda_runtime.h>

void cudaMemcpyDataArray(DataArray destinationDataArray, DataArray sourceDataArray, cudaMemcpyKind kind);

void cudaMemcpyFirstDataArray(DataArray destinationDataArray, DataArray sourceDataArray, int particleCount);

#endif	/* DATA_CUH */

