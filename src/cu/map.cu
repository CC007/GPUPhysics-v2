/* 
 * File:   map.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:09 PM
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../../include/map.h"
#include "../../include/safemem.h"
#include "../../include/extendedio.h"

void cudaMallocMap(Map *mapPointer, int rowCount) {
	Map hostHelperMap;
	Map devHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _Map))) {
		eprintf("The map could not be allocated on the host");
	}
	if (safeCudaMalloc((void**) &devHelperMap, 1, sizeof (struct _Map))) {
		eprintf("The map could not be allocated on the device");
	}
	hostHelperMap->length = rowCount;
	if (rowCount > 0) {
		int mallocFailed = 0;
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->A), rowCount, sizeof (double));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->x), rowCount, sizeof (int));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->dx), rowCount, sizeof (int));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->y), rowCount, sizeof (int));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->dy), rowCount, sizeof (int));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->delta), rowCount, sizeof (int));
		mallocFailed += safeCudaCalloc((void**) &(hostHelperMap->phi), rowCount, sizeof (int));
		if (mallocFailed) {
			eprintf("The map's contents could not be allocated");
		}
		if (safeCudaMemcpyHtD(devHelperMap, hostHelperMap, 1, sizeof (struct _Map))) {
			eprintf("The map's contents could not be made available in device memory (temporary host map memcpy failed)");
		}
		if (safeFree((void**) &hostHelperMap)) {
			wprintf("The temporary host map's contents could not be freed");
		}
		*mapPointer = devHelperMap;
	}
}

void cudaFreeMap(Map *mapPointer) {
	Map devHelperMap = *mapPointer;
	Map hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _Map))) {
		wprintf("The map's contents could not be accessed (temporary host map alloc failed)");
		return;
	}
	if (safeCudaMemcpyDtH(&hostHelperMap, &devHelperMap, 1, sizeof (struct _Map))) {
		wprintf("The map's contents could not be accessed (temporary host map memcpy failed)");
		return;
	}
	int freeFailed = 0;
	if (hostHelperMap->length > 0) {
		freeFailed += safeCudaFree((void**) &(hostHelperMap->A));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->x));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->dx));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->y));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->dy));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->delta));
		freeFailed += safeCudaFree((void**) &(hostHelperMap->phi));
	}
	if (freeFailed) {
		wprintf("The map's contents could not be freed");
	}
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host map's contents could not be freed");
	}
	if (safeCudaFree((void**) mapPointer)) {
		wprintf("The map could not be freed");
	}
}

void cudaMemcpyMap(Map destinationMap, Map sourceMap, cudaMemcpyKind kind) {
	Map hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _Map))) {
		eprintf("The map could not be copied (temporary host map alloc failed)");
	}
	int memcpyFailed = 0;
	if (kind == cudaMemcpyDeviceToHost) {
		if (safeCudaMemcpyDtH(hostHelperMap, sourceMap, 1, sizeof (struct _Map))) {
			eprintf("The map could not be copied (source map pointers couldn't be accessed)");
		}
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->A, hostHelperMap->A, hostHelperMap->length, sizeof (double));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->x, hostHelperMap->x, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->dx, hostHelperMap->dx, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->y, hostHelperMap->y, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->dy, hostHelperMap->dy, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->delta, hostHelperMap->delta, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyDtH(destinationMap->phi, hostHelperMap->phi, hostHelperMap->length, sizeof (int));
	} else if (kind == cudaMemcpyHostToDevice) {
		if (safeCudaMemcpyDtH(hostHelperMap, destinationMap, 1, sizeof (struct _Map))) {
			eprintf("The map could not be copied (destination map pointers couldn't be accessed)");
		}
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->A, sourceMap->A, hostHelperMap->length,sizeof (double));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->x, sourceMap->x, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->dx, sourceMap->dx, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->y, sourceMap->y, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->dy, sourceMap->dy, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->delta, sourceMap->delta, hostHelperMap->length, sizeof (int));
		memcpyFailed += safeCudaMemcpyHtD(hostHelperMap->phi, sourceMap->phi, hostHelperMap->length, sizeof (int));
	} else {
		eprintf("DeviceToDevice is not yet supported for maps!\n");
	}
	if(memcpyFailed){
			eprintf("The map could not be copied (copying the content of the map failed)");
	}
}