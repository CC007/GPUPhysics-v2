/* 
 * File:   map.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on November 5, 2015, 9:47 PM
 */

#include <stdlib.h>
#include <stdio.h>

#include "../../include/spinmap.h"
#include "../../include/safemem.h"
#include "../../include/extendedio.h"

/* Allocate memory for a inner spin map with rowCount rows
 *  
 *  \param innerSpinMapPointer - a pointer to the inner spin map that needs to be allocated
 *  \param rowCount - the number of rows that the maps in the inner spin map consists of
 */
void mallocInnerSpinMap(InnerSpinMap *innerSpinMapPointer, int rowCount) {
    InnerSpinMap helperMap;
    if (safeMalloc((void**) &helperMap, 1, sizeof (struct _InnerSpinMap))) {
        eprintf("The inner spin map could not be allocated\n");
    }
	mallocMap(&(helperMap->x), rowCount);
	mallocMap(&(helperMap->y), rowCount);
	mallocMap(&(helperMap->z), rowCount);
	
    *innerSpinMapPointer = helperMap;
}

/* Free the memory of the provided inner spin map 
 *  
 *  \param innerSpinMapPointer - a pointer to the inner spin map that needs to be freed
 */
void freeInnerSpinMap(InnerSpinMap *innerSpinMapPointer) {
    InnerSpinMap helperMap = *innerSpinMapPointer;
	
    freeMap(&(helperMap->x));
	freeMap(&(helperMap->y));
	freeMap(&(helperMap->z));
	
    if (safeFree((void**) innerSpinMapPointer)) {
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
void mallocSpinMap(SpinMap *spinMapPointer, int rowCountX, int rowCountY, int rowCountZ) {
    SpinMap helperMap;
    if (safeMalloc((void**) &helperMap, 1, sizeof (struct _SpinMap))) {
        eprintf("The spin map could not be allocated\n");
    }
	mallocInnerSpinMap(&(helperMap->x), rowCountX);
	mallocInnerSpinMap(&(helperMap->y), rowCountY);
	mallocInnerSpinMap(&(helperMap->z), rowCountZ);
    
    *spinMapPointer = helperMap;
}

/* Free the memory of the provided spin map 
 *  
 *  \param spinMapPointer - a pointer to the spin map that needs to be freed
 */
void freeSpinMap(SpinMap *spinMapPointer) {
    SpinMap helperMap = *spinMapPointer;
	
    freeInnerSpinMap(&(helperMap->x));
	freeInnerSpinMap(&(helperMap->y));
	freeInnerSpinMap(&(helperMap->z));
	
    if (safeFree((void**) spinMapPointer)) {
        wprintf("The spin map could not be freed\n");
    }
}
