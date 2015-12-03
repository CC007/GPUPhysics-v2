/* 
 * File:   data.c
 * Author: Rik Schaaf aka CC007 (coolcat007.nl)
 *
 * Created on April 27, 2015, 7:54 PM
 */

#include <stdlib.h>
#include <stdio.h>

#include "../../include/data.h"
#include "../../include/safemem.h"
#include "../../include/extendedio.h"

void mallocDataArray(DataArray *dataArray, int iterationCount, int particleCount) {
	DataArray helperDataArray;
	if (safeMalloc((void**) &helperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated\n");
	}
	for (int i = 0; i < particleCount; i++) {
		helperDataArray[i].length = iterationCount;
	}
	if (iterationCount > 0) {
		int mallocFailed = 0;
		int i;
		for (i = 0; i < particleCount && !mallocFailed; i++) {
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].x), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].dx), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].y), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].dy), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].delta), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].phi), iterationCount, sizeof (double));
		}
		if (mallocFailed) {
			eprintf("The data array's contents could not be allocated (at particle %d)\n", i);
		}
	}
	*dataArray = helperDataArray;
}

void freeDataArray(DataArray *dataArray, int particleCount) {
	DataArray helperDataArray = *dataArray;
	int freeFailed = 0;
	if (helperDataArray->length > 0) {
		for (int i = 0; i < particleCount && !freeFailed; i++) {
			freeFailed += safeFree((void**) &(helperDataArray[i].x));
			freeFailed += safeFree((void**) &(helperDataArray[i].dx));
			freeFailed += safeFree((void**) &(helperDataArray[i].y));
			freeFailed += safeFree((void**) &(helperDataArray[i].dy));
			freeFailed += safeFree((void**) &(helperDataArray[i].delta));
			freeFailed += safeFree((void**) &(helperDataArray[i].phi));
		}
	}
	if (freeFailed) {
		wprintf("The data array's contents could not be freed\n");
	}
	if (safeFree((void**) dataArray)) {
		wprintf("The data array could not be freed\n");
	}
}

void mallocSpinDataArray(SpinDataArray *spinDataArray, int iterationCount, int particleCount) {
	SpinDataArray helperSpinDataArray;
	if (safeMalloc((void**) &helperSpinDataArray, particleCount, sizeof (struct _SpinDataArray))) {
		eprintf("The spin data array could not be allocated\n");
	}
	for (int i = 0; i < particleCount; i++) {
		helperSpinDataArray[i].length = iterationCount;
	}
	if (iterationCount > 0) {
		int mallocFailed = 0;
		int i;
		for (i = 0; i < particleCount && !mallocFailed; i++) {
			mallocFailed += safeCalloc((void**) &(helperSpinDataArray[i].sx), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperSpinDataArray[i].sy), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperSpinDataArray[i].sz), iterationCount, sizeof (double));
		}
		if (mallocFailed) {
			eprintf("The spin data array's contents could not be allocated (at particle %d)\n", i);
		}
	}
	*spinDataArray = helperSpinDataArray;
}

void freeSpinDataArray(SpinDataArray *spinDataArray, int particleCount) {
	SpinDataArray helperSpinDataArray = *spinDataArray;
	int freeFailed = 0;
	if (helperSpinDataArray->length > 0) {
		for (int i = 0; i < particleCount && !freeFailed; i++) {
			freeFailed += safeFree((void**) &(helperSpinDataArray[i].sx));
			freeFailed += safeFree((void**) &(helperSpinDataArray[i].sy));
			freeFailed += safeFree((void**) &(helperSpinDataArray[i].sz));
		}
	}
	if (freeFailed) {
		wprintf("The spin data array's contents could not be freed\n");
	}
	if (safeFree((void**) spinDataArray)) {
		wprintf("The spin data array could not be freed\n");
	}
}
