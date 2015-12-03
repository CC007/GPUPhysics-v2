/* 
 * File:   spinmap.h
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on November 5, 2015, 9:46 AM
 */

#ifndef SPINMAP_H
#define	SPINMAP_H

#include "map.h"

#ifdef	__cplusplus
extern "C" {
#endif

	typedef struct _InnerSpinMap {
		Map x;
		Map y;
		Map z;
	} *InnerSpinMap;

	void mallocInnerSpinMap(InnerSpinMap *innerSpinMapPointer, int rowCount);
	
	void freeInnerSpinMap(InnerSpinMap *innerSpinMapPointer);
	
	void cudaMallocInnerSpinMap(InnerSpinMap *innerSpinMapPointer, int rowCount);
	
	void cudaFreeInnerSpinMap(InnerSpinMap *innerSpinMapPointer);

	typedef struct _SpinMap {
		InnerSpinMap x;
		InnerSpinMap y;
		InnerSpinMap z;
	} *SpinMap;

	void mallocSpinMap(SpinMap *spinMapPointer, int rowXCount, int rowYCount, int rowZCount);
	
	void freeSpinMap(SpinMap *spinMapPointer);
	
	void cudaMallocSpinMap(SpinMap *spinMapPointer, int rowXCount, int rowYCount, int rowZCount);
	
	void cudaFreeSpinMap(SpinMap *spinMapPointer);

#ifdef	__cplusplus
}
#endif

#endif	/* SPINMAP_H */

