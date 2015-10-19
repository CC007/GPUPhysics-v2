/* 
 * File:   data.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:03 PM
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/data.h"
#include "../../include/safemem.h"
#include "../../include/safemem.cuh"
#include "../../include/extendedio.h"

void cudaMallocDataArray(DataArray *dataArray, int iterationCount, int particleCount) {
	DataArray hostHelperDataArray;
	DataArray devHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated on the host\n");
	}
	if (safeCudaMalloc((void**) &devHelperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated on the device\n");
	}
	for (int i = 0; i < particleCount; i++) {
		hostHelperDataArray[i].length = iterationCount;
	}
	if (iterationCount > 0) {
		int mallocFailed = 0;
		for (int i = 0; i < particleCount; i++) {
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].x), iterationCount, sizeof (double));
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].dx), iterationCount, sizeof (double));
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].y), iterationCount, sizeof (double));
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].dy), iterationCount, sizeof (double));
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].delta), iterationCount, sizeof (double));
			mallocFailed += safeCudaCalloc((void**) &(hostHelperDataArray[i].phi), iterationCount, sizeof (double));
		}
		if (mallocFailed) {
			eprintf("The data array's contents could not be allocated\n");
		}
		if (safeCudaMemcpyHtD(devHelperDataArray, hostHelperDataArray, particleCount, sizeof (struct _DataArray))) {
			eprintf("The data array's contents could not be made available in device memory (temporary host data array memcpy failed)\n");
		}
		if (safeFree((void**) &hostHelperDataArray)) {
			wprintf("The temporary host data array's contents could not be freed\n");
		}
		*dataArray = devHelperDataArray;
	}
}

void cudaFreeDataArray(DataArray *dataArray, int particleCount) {
	DataArray devHelperDataArray = *dataArray;
	DataArray hostHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
		wprintf("The data array could not be freed (temporary host data array alloc failed)\n");
		return;
	}
	int freeFailed = 0;
	if (hostHelperDataArray->length > 0) {
		for (int i = 0; i < particleCount; i++) {
			if (safeCudaMemcpyDtH(hostHelperDataArray, &(devHelperDataArray[i]), 1, sizeof (struct _DataArray))) {
				wprintf("The data array could not be freed (temporary host data array memcpy failed)\n");
				return;
			}
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->x));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->dx));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->y));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->dy));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->delta));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray->phi));
		}
	}
	if (freeFailed) {
		wprintf("The map's contents could not be freed\n");
	}
	if (safeFree((void**) &hostHelperDataArray)) {
		wprintf("The temporary host map's contents could not be freed\n");
	}
	if (safeCudaFree((void**) dataArray)) {
		wprintf("The map could not be freed\n");
	}
}

void cudaMemcpyDataArray(DataArray destinationDataArray, DataArray sourceDataArray, cudaMemcpyKind kind) {
	DataArray hostHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
		eprintf("The map could not be copied (temporary host data array alloc failed)\n");
	}
	int memcpyFailed = 0;
	if (kind == cudaMemcpyDeviceToHost) {
		if (safeCudaMemcpyDtH(hostHelperDataArray, sourceDataArray, 1, sizeof (struct _DataArray))) {
			eprintf("The data array could not be copied (source data array pointers couldn't be accessed)\n");
		}
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->x, hostHelperDataArray->x, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->dx, hostHelperDataArray->dx, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->y, hostHelperDataArray->y, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->dy, hostHelperDataArray->dy, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->delta, hostHelperDataArray->delta, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->phi, hostHelperDataArray->phi, hostHelperDataArray->length, sizeof (double));
	} else if (kind == cudaMemcpyHostToDevice) {
		if (safeCudaMemcpyDtH(hostHelperDataArray, destinationDataArray, 1, sizeof (struct _DataArray))) {
			eprintf("The data array could not be copied (source data array pointers couldn't be accessed)");
		}
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->x, sourceDataArray->x, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->dx, sourceDataArray->dx, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->y, sourceDataArray->y, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->dy, sourceDataArray->dy, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->delta, sourceDataArray->delta, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->phi, sourceDataArray->phi, hostHelperDataArray->length, sizeof (double));
	} else {
		eprintf("DeviceToDevice is not yet supported for data arrays!\n");
	}
	if (memcpyFailed) {
		eprintf("The data array could not be copied (copying the content of the data array failed)\n");
	}
	if (safeFree((void**) &hostHelperDataArray)) {
		wprintf("The temporary host data array's contents could not be freed\n");
	}
}

void cudaMemcpyFirstDataArray(DataArray destinationDataArray, DataArray sourceDataArray, int particleCount) {
	DataArray hostHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
		eprintf("The map could not be copied (temporary host data array alloc failed)\n");
	}
	int memcpyFailed = 0;
	for (int i = 0; i < particleCount; i++) {
		if (safeCudaMemcpyDtH(hostHelperDataArray, &(destinationDataArray[i]), 1, sizeof (struct _DataArray))) {
			eprintf("The data array could not be copied (source map pointers couldn't be accessed)\n");
		}
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->x, sourceDataArray[i].x, 1, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->dx, sourceDataArray[i].dx, 1, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->y, sourceDataArray[i].y, 1, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->dy, sourceDataArray[i].dy, 1, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->delta, sourceDataArray[i].delta, 1, sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperDataArray->phi, sourceDataArray[i].phi, 1, sizeof (double));
	}
	if (memcpyFailed) {
		eprintf("The data array could not be copied (copying the content of the data array failed)\n");
	}
	if (safeFree((void**) &hostHelperDataArray)) {
		wprintf("The temporary host data array's contents could not be freed\n");
	}
}
