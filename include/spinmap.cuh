/* 
 * File:   spinmap.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on December 1, 2015, 5:12 PM
 */

#ifndef SPINMAP_CUH
#define	SPINMAP_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef	__cplusplus

void cudaMemcpyInnerSpinMap(InnerSpinMap destinationInnerSpinMap, InnerSpinMap sourceInnerSpinMap, cudaMemcpyKind kind);

void cudaMemcpySpinMap(SpinMap destinationSpinMap, SpinMap sourceSpinMap, cudaMemcpyKind kind);

#endif  /* __cplusplus */

#endif	/* SPINMAP_CUH */

