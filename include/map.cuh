/* 
 * File:   map.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on October 13, 2015, 6:10 AM
 */

#ifndef MAP_CUH
#define	MAP_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef	__cplusplus

void cudaMemcpyMap(Map destinationMap, Map sourceMap, cudaMemcpyKind kind);

#endif  /* __cplusplus */

#endif	/* MAP_CUH */

