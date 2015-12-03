/* 
 * File:   map.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on November 5, 2015, 9:47 PM
 */

#include <stdlib.h>

#include "../../include/map.h"
#include "../../include/map.cuh"
#include "../../include/spinmap.h"
#include "../../include/spinmap.cuh"
#include "../../include/safemem.h"
#include "../../include/extendedio.h"

/* Allocate memory for a inner spin map with rowCount rows
 *  
 *  \param innerSpinMapPointer - a pointer to the inner spin map that needs to be allocated
 *  \param rowCount - the number of rows that the maps in the inner spin map consists of
 */
void cudaMallocInnerSpinMap(InnerSpinMap *innerSpinMapPointer, int rowCount) {
	InnerSpinMap devHelperMap;
	InnerSpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		eprintf("The inner spin map could not be allocated on the host\n");
	}
	if (safeCudaMalloc((void**) &devHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		eprintf("The inner spin map could not be allocated on the device\n");
	}
	cudaMallocMap(&(hostHelperMap->x), rowCount);
	cudaMallocMap(&(hostHelperMap->y), rowCount);
	cudaMallocMap(&(hostHelperMap->z), rowCount);

	if (safeCudaMemcpyHtD(devHelperMap, hostHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		eprintf("The inner spin map's contents could not be made available in device memory (temporary host map memcpy failed)\n");
	}
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host inner spin map's contents could not be freed\n");
	}
	*innerSpinMapPointer = devHelperMap;
}

/* Free the memory of the provided inner spin map 
 *  
 *  \param innerSpinMapPointer - a pointer to the inner spin map that needs to be freed
 */
void cudaFreeInnerSpinMap(InnerSpinMap *innerSpinMapPointer) {
	InnerSpinMap devHelperMap = *innerSpinMapPointer;
	InnerSpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		wprintf("The inner spin map's contents could not be accessed (temporary host map alloc failed)\n");
		return;
	}
	if (safeCudaMemcpyDtH(hostHelperMap, devHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		wprintf("The inner spin map's contents could not be accessed (temporary host map memcpy failed)\n");
		return;
	}

	cudaFreeMap(&(hostHelperMap->x));
	cudaFreeMap(&(hostHelperMap->y));
	cudaFreeMap(&(hostHelperMap->z));
	
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host inner spin map's contents could not be freed\n");
	}
	if (safeCudaFree((void**) innerSpinMapPointer)) {
		wprintf("The inner spin map could not be freed\n");
	}
}

/* Allocate memory for a spin map with rowCountX, Y and Z rows
 *  
 *  \param spinMapPointer - a pointer to the spin map that needs to be allocated
 *  \param rowCountX - the number of rows that the x spin in the spin map consists of
 *  \param rowCountY - the number of rows that the y spin in the spin map consists of
 *  \param rowCountZ - the number of rows that the z spin in the spin map consists of
 */
void cudaMallocSpinMap(SpinMap *spinMapPointer, int rowCountX, int rowCountY, int rowCountZ) { 
	SpinMap devHelperMap;
	SpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _SpinMap))) {
		eprintf("The spin map could not be allocated\n");
	}
	if (safeCudaMalloc((void**) &devHelperMap, 1, sizeof (struct _SpinMap))) {
		eprintf("The spin map could not be allocated on the device\n");
	}
	cudaMallocInnerSpinMap(&(hostHelperMap->x), rowCountX);
	cudaMallocInnerSpinMap(&(hostHelperMap->y), rowCountY);
	cudaMallocInnerSpinMap(&(hostHelperMap->z), rowCountZ);

	if (safeCudaMemcpyHtD(devHelperMap, hostHelperMap, 1, sizeof (struct _SpinMap))) {
		eprintf("The spin map's contents could not be made available in device memory (temporary host map memcpy failed)\n");
	}
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host spin map's contents could not be freed\n");
	}

	*spinMapPointer = devHelperMap;
}

/* Free the memory of the provided spin map 
 *  
 *  \param spinMapPointer - a pointer to the spin map that needs to be freed
 */
void cudaFreeSpinMap(SpinMap *spinMapPointer) {
	SpinMap devHelperMap = *spinMapPointer;
	SpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _SpinMap))) {
		wprintf("The spin map's contents could not be accessed (temporary host map alloc failed)\n");
		return;
	}
	if (safeCudaMemcpyDtH(hostHelperMap, devHelperMap, 1, sizeof (struct _SpinMap))) {
		wprintf("The spin map's contents could not be accessed (temporary host map memcpy failed)\n");
		return;
	}

	cudaFreeInnerSpinMap(&(hostHelperMap->x));
	cudaFreeInnerSpinMap(&(hostHelperMap->y));
	cudaFreeInnerSpinMap(&(hostHelperMap->z));
	
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host spin map's contents could not be freed\n");
	}
	if (safeFree((void**) spinMapPointer)) {
		wprintf("The spin map could not be freed\n");
	}
}

void cudaMemcpyInnerSpinMap(InnerSpinMap destinationInnerSpinMap, InnerSpinMap sourceInnerSpinMap, cudaMemcpyKind kind){
	InnerSpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _InnerSpinMap))) {
		eprintf("The inner spin map could not be copied (temporary host spin map alloc failed)\n");
	}
	if (kind == cudaMemcpyDeviceToHost) {
		if (safeCudaMemcpyDtH(hostHelperMap, sourceInnerSpinMap, 1, sizeof (struct _InnerSpinMap))) {
			eprintf("The inner spin map could not be copied (source inner spin map pointers couldn't be accessed)\n");
		}
		cudaMemcpyMap(destinationInnerSpinMap->x, hostHelperMap->x, kind);
		cudaMemcpyMap(destinationInnerSpinMap->y, hostHelperMap->y, kind);
		cudaMemcpyMap(destinationInnerSpinMap->z, hostHelperMap->z, kind);
	} else if (kind == cudaMemcpyHostToDevice) {
		if (safeCudaMemcpyDtH(hostHelperMap, destinationInnerSpinMap, 1, sizeof (struct _InnerSpinMap))) {
			eprintf("The inner spin map could not be copied (destination inner spin map pointers couldn't be accessed)");
		}
		cudaMemcpyMap(hostHelperMap->x, sourceInnerSpinMap->x, kind);
		cudaMemcpyMap(hostHelperMap->y, sourceInnerSpinMap->y, kind);
		cudaMemcpyMap(hostHelperMap->z, sourceInnerSpinMap->z, kind);
	} else {
		eprintf("DeviceToDevice is not yet supported for inner spin maps!\n");
	}
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host inner spin map's contents could not be freed\n");
	}
}
	

void cudaMemcpySpinMap(SpinMap destinationSpinMap, SpinMap sourceSpinMap, cudaMemcpyKind kind){
	SpinMap hostHelperMap;
	if (safeMalloc((void**) &hostHelperMap, 1, sizeof (struct _SpinMap))) {
		eprintf("The spin map could not be copied (temporary host spin map alloc failed)\n");
	}
	if (kind == cudaMemcpyDeviceToHost) {
		if (safeCudaMemcpyDtH(hostHelperMap, sourceSpinMap, 1, sizeof (struct _SpinMap))) {
			eprintf("The spin map could not be copied (source spin map pointers couldn't be accessed)\n");
		}
		cudaMemcpyInnerSpinMap(destinationSpinMap->x, hostHelperMap->x, kind);
		cudaMemcpyInnerSpinMap(destinationSpinMap->y, hostHelperMap->y, kind);
		cudaMemcpyInnerSpinMap(destinationSpinMap->z, hostHelperMap->z, kind);
	} else if (kind == cudaMemcpyHostToDevice) {
		if (safeCudaMemcpyDtH(hostHelperMap, destinationSpinMap, 1, sizeof (struct _SpinMap))) {
			eprintf("The spin map could not be copied (destination spin map pointers couldn't be accessed)");
		}
		cudaMemcpyInnerSpinMap(hostHelperMap->x, sourceSpinMap->x, kind);
		cudaMemcpyInnerSpinMap(hostHelperMap->y, sourceSpinMap->y, kind);
		cudaMemcpyInnerSpinMap(hostHelperMap->z, sourceSpinMap->z, kind);
	} else {
		eprintf("DeviceToDevice is not yet supported for spin maps!\n");
	}
	if (safeFree((void**) &hostHelperMap)) {
		wprintf("The temporary host spin map's contents could not be freed\n");
	}
}
