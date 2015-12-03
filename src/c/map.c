/* 
 * File:   map.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on April 27, 2015, 7:33 PM
 */

#include <stdlib.h>
#include <stdio.h>

#include "../../include/map.h"
#include "../../include/safemem.h"
#include "../../include/extendedio.h"

/* Allocate memory for a map with rowCount rows
 *  
 *  \param mapPointer - a pointer to the map that needs to be allocated
 *  \param rowCount - the number of rows that the map consists of
 */
void mallocMap(Map *mapPointer, int rowCount) {
    Map helperMap;
    if (safeMalloc((void**) &helperMap, 1, sizeof (struct _Map))) {
        eprintf("The map could not be allocated\n");
    }
    helperMap->length = rowCount;
    if (rowCount > 0) {
        int mallocFailed = 0;
        mallocFailed += safeCalloc((void**) &(helperMap->A), rowCount, sizeof (double));
        mallocFailed += safeCalloc((void**) &(helperMap->x), rowCount, sizeof (int));
        mallocFailed += safeCalloc((void**) &(helperMap->dx), rowCount, sizeof (int));
        mallocFailed += safeCalloc((void**) &(helperMap->y), rowCount, sizeof (int));
        mallocFailed += safeCalloc((void**) &(helperMap->dy), rowCount, sizeof (int));
        mallocFailed += safeCalloc((void**) &(helperMap->delta), rowCount, sizeof (int));
        mallocFailed += safeCalloc((void**) &(helperMap->phi), rowCount, sizeof (int));
        if (mallocFailed) {
            eprintf("The map's contents could not be allocated\n");
        }
    }
    *mapPointer = helperMap;
}

/* Free the memory of the provided map 
 *  
 *  \param mapPointer - a pointer to the map that needs to be freed
 */
void freeMap(Map *mapPointer) {
    Map helperMap = *mapPointer;
    int freeFailed = 0;
    if (helperMap->length > 0) {
        freeFailed += safeFree((void**) &(helperMap->A));
        freeFailed += safeFree((void**) &(helperMap->x));
        freeFailed += safeFree((void**) &(helperMap->dx));
        freeFailed += safeFree((void**) &(helperMap->y));
        freeFailed += safeFree((void**) &(helperMap->dy));
        freeFailed += safeFree((void**) &(helperMap->delta));
        freeFailed += safeFree((void**) &(helperMap->phi));
    }
    if (freeFailed) {
        wprintf("The map's contents could not be freed\n");
    }
    if (safeFree((void**) mapPointer)) {
        wprintf("The map could not be freed\n");
    }
}