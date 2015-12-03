/* 
 * File:   kernel.h
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on Oktober 4, 2015, 4:38 PM
 */

#ifndef KERNEL_H
#define	KERNEL_H

#include "data.h"
#include "map.h"
#include "spinmap.h"

#ifdef	__cplusplus
extern "C" {
#endif

	void kernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount);
	
	void spinKernel(DataArray dataArray, SpinDataArray spinDataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, SpinMap spinMap, int particleCount, int iterationCount);


#ifdef	__cplusplus
}
#endif

#endif	/* KERNEL_H */

