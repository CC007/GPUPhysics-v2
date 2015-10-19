/* 
 * File:   kernel.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on Oktober 4, 2015, 4:38 PM
 */

#include <math.h>

#include "../../include/safemem.h"
#include "../../include/extendedio.h"
#include "../../include/data.h"
#include "../../include/map.h"

void sumArrayHelper(double *nums, int length, int interval) {
    int index = 0;
    int next = interval / 2;
    do {
        if (next < length) {
            nums[index] += nums[next];
        }
        index += interval;
        next += interval;
    } while (index < length);
}

double sumArray(double *nums, int length) {
    if (length <= 0) {
        return 0;
    }
    int interval = 2;
    while (interval < length * 2) {
        sumArrayHelper(nums, length, interval);
        interval *= 2;
    }
    return nums[0];
}

void calcData(DataArray dataArray, int iteration, Map map, double *newValue) {
    double *nums;
	if(safeCalloc((void**)&nums, map->length, sizeof (double))){
		eprintf("Unable to to calculate iteration %d (local variable for the calculation could not be allocated", iteration);
	}

    for (int i = 0; i < map->length; i++) {
        nums[i] = map->A[i] 
				* pow(dataArray->x[iteration], map->x[i])
                * pow(dataArray->dx[iteration], map->dx[i])
                * pow(dataArray->y[iteration], map->y[i])
                * pow(dataArray->dy[iteration], map->dy[i])
                * pow(dataArray->delta[iteration], map->delta[i])
                * pow(dataArray->phi[iteration], map->phi[i]);
    }

    *newValue = sumArray(nums, map->length);
    if(safeFree((void**)&nums)){
		wprintf("Unable to free local variable");
	}
}


void kernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount) {
    for (int n = 0; n < particleCount; n++) {
        for (int i = 0; i < iterationCount - 1; i++) {
            calcData(&(dataArray[n]), i, x, &(dataArray[n].x[i + 1]));
            calcData(&(dataArray[n]), i, dx, &(dataArray[n].dx[i + 1]));
            calcData(&(dataArray[n]), i, y, &(dataArray[n].y[i + 1]));
            calcData(&(dataArray[n]), i, dy, &(dataArray[n].dy[i + 1]));
            calcData(&(dataArray[n]), i, delta, &(dataArray[n].delta[i + 1]));
            calcData(&(dataArray[n]), i, phi, &(dataArray[n].phi[i + 1]));
        }
    }
}
