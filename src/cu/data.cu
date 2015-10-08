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

void cudaMallocData(DataArray *dataArray, int iterationCount, int particleCount) {
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

void cudaMemcpyMap(Map *dst_m, Map *src_m, cudaMemcpyKind kind) {
	Map helper_m;
	if (kind == cudaMemcpyDeviceToHost) {
		cudaMemcpy(&helper_m, src_m, sizeof (Map), cudaMemcpyDeviceToHost);
		cudaMemcpy(dst_m->A, helper_m.A, helper_m.length * sizeof (double), kind);
		cudaMemcpy(dst_m->x, helper_m.x, helper_m.length * sizeof (int), kind);
		cudaMemcpy(dst_m->dx, helper_m.dx, helper_m.length * sizeof (int), kind);
		cudaMemcpy(dst_m->y, helper_m.y, helper_m.length * sizeof (int), kind);
		cudaMemcpy(dst_m->dy, helper_m.dy, helper_m.length * sizeof (int), kind);
		cudaMemcpy(dst_m->delta, helper_m.delta, helper_m.length * sizeof (int), kind);
		cudaMemcpy(dst_m->phi, helper_m.phi, helper_m.length * sizeof (int), kind);
	} else if (kind == cudaMemcpyHostToDevice) {
		cudaMemcpy(&helper_m, dst_m, sizeof (Map), cudaMemcpyDeviceToHost);
		cudaMemcpy(helper_m.A, src_m->A, helper_m.length * sizeof (double), kind);
		cudaMemcpy(helper_m.x, src_m->x, helper_m.length * sizeof (int), kind);
		cudaMemcpy(helper_m.dx, src_m->dx, helper_m.length * sizeof (int), kind);
		cudaMemcpy(helper_m.y, src_m->y, helper_m.length * sizeof (int), kind);
		cudaMemcpy(helper_m.dy, src_m->dy, helper_m.length * sizeof (int), kind);
		cudaMemcpy(helper_m.delta, src_m->delta, helper_m.length * sizeof (int), kind);
		cudaMemcpy(helper_m.phi, src_m->phi, helper_m.length * sizeof (int), kind);
	} else {
		fprintf(stderr, "DeviceToDevice is not yet supported for maps!\n");
		getchar();
		exit(EXIT_FAILURE);
	}

}