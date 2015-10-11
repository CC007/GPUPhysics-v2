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
#include "../../include/extendedio.h"

void cudaMallocDataArray(DataArray *dataArray, int iterationCount, int particleCount) {
	DataArray hostHelperDataArray;
	DataArray devHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated on the host");
	}
	if (safeCudaMalloc((void**) &devHelperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated on the device");
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
			eprintf("The data array's contents could not be allocated");
		}
		if (safeCudaMemcpyHtD(&devHelperDataArray, &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
			eprintf("The data array's contents could not be made available in device memory (temporary host data array memcpy failed)");
		}
		if (safeFree((void**) &hostHelperDataArray)) {
			wprintf("The temporary host data array's contents could not be freed");
		}
		*dataArray = devHelperDataArray;
	}
}

void cudaFreeDataArray(DataArray *dataArray, int particleCount) {
	DataArray devHelperDataArray = *dataArray;
	DataArray hostHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
		wprintf("The map's contents could not be accessed (temporary host map alloc failed)");
		return;
	}
	if (safeCudaMemcpyDtH(&hostHelperDataArray, &devHelperDataArray, 1, sizeof (struct _DataArray))) {
		wprintf("The map's contents could not be accessed (temporary host map memcpy failed)");
		return;
	}
	int freeFailed = 0;
	if (hostHelperDataArray->length > 0) {
		for (int i = 0; i < particleCount; i++) {
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].x));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].dx));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].y));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].dy));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].delta));
			freeFailed += safeCudaFree((void**) &(hostHelperDataArray[i].phi));
		}
	}
	if (freeFailed) {
		wprintf("The map's contents could not be freed");
	}
	if (safeFree((void**) &hostHelperDataArray)) {
		wprintf("The temporary host map's contents could not be freed");
	}
	if (safeCudaFree((void**) dataArray)) {
		wprintf("The map could not be freed");
	}
}

void cudaMemcpyDataArray(DataArray destinationDataArray, DataArray sourceDataArray, cudaMemcpyKind kind) {
	DataArray hostHelperDataArray;
	if (safeMalloc((void**) &hostHelperDataArray, 1, sizeof (struct _DataArray))) {
		eprintf("The map could not be copied (temporary host map alloc failed)");
	}
	int memcpyFailed = 0;
	if (kind == cudaMemcpyDeviceToHost) {
		if (safeCudaMemcpyDtH(hostHelperDataArray, sourceDataArray, 1, sizeof (struct _DataArray))) {
			eprintf("The data array could not be copied (source map pointers couldn't be accessed)");
		}
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->x, hostHelperDataArray->x, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->dx, hostHelperDataArray->dx, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->y, hostHelperDataArray->y, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->dy, hostHelperDataArray->dy, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->delta, hostHelperDataArray->delta, hostHelperDataArray->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationDataArray->phi, hostHelperDataArray->phi, hostHelperDataArray->length, sizeof (double));
	} else if (kind == cudaMemcpyHostToDevice) {
		if (safeCudaMemcpyDtH(hostHelperDataArray, sourceDataArray, 1, sizeof (struct _DataArray))) {
			eprintf("The data array could not be copied (source map pointers couldn't be accessed)");
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
	if(memcpyFailed){
			eprintf("The data array could not be copied (copying the content of the data array failed)");
	}

}