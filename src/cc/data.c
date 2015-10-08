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

void mallocData(DataArray *dataArray, int iterationCount, int particleCount) {
	DataArray helperDataArray;
	if (safeMalloc((void**) &helperDataArray, particleCount, sizeof (struct _DataArray))) {
		eprintf("The data array could not be allocated");
	}
	for (int i = 0; i < particleCount; i++) {
		helperDataArray[i].length = iterationCount;
	}
	if (iterationCount > 0) {
		int mallocFailed = 0;
		for (int i = 0; i < particleCount; i++) {
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].x), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].dx), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].y), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].dy), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].delta), iterationCount, sizeof (double));
			mallocFailed += safeCalloc((void**) &(helperDataArray[i].phi), iterationCount, sizeof (double));
		}
		if (mallocFailed) {
			eprintf("The data array's contents could not be allocated");
		}
	}
	*dataArray = helperDataArray;
}

void freeData(DataArray *dataArray, int particleCount) {
	DataArray helperDataArray = *dataArray;
	int freeFailed = 0;
	if (helperDataArray->length > 0) {
		for (int i = 0; i < particleCount; i++) {
			freeFailed += safeFree((void**) &(helperDataArray[i].x));
			freeFailed += safeFree((void**) &(helperDataArray[i].dx));
			freeFailed += safeFree((void**) &(helperDataArray[i].y));
			freeFailed += safeFree((void**) &(helperDataArray[i].dy));
			freeFailed += safeFree((void**) &(helperDataArray[i].delta));
			freeFailed += safeFree((void**) &(helperDataArray[i].phi));
		}
	}
	if (freeFailed) {
		wprintf("The data array's contents could not be freed");
	}
	if (safeFree((void**) dataArray)) {
		wprintf("The data array could not be freed");
	}
}
