/* 
 * File:   safemem.cuh
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on 13 oktober 2015, 0:33
 */

#ifndef SAFEMEM_CUH
#define	SAFEMEM_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "safememstates.h"

/* Allocate elemCount * elemSize bytes of device memory from a kernel
 * The program will check if the memory was successfully allocated
 * 
 *  \param pp - a pointer to the pointer that afterwards points to the allocated memory
 *  \param elemCount - the number of elements in the newly allocated memory block
 *  \param elemSize - the size (in bytes) of each element in the newly allocated memory block
 * 
 *  \return
 *  ALLOC_SUCCESS - the memory is successfully allocated
 *    ALLOC_FAILURE - the memory allocation was unsuccessful
 */
__device__ int safeDeviceMalloc(void **pp, int elemCount, int elemSize);

/* Allocate elemCount * elemSize bytes of device memory from a kernel, all elements initialized to 0
 * The program will check if the memory was successfully allocated
 * 
 *  \param pp - a pointer to the pointer that afterwards points to the allocated memory
 *  \param elemCount - the number of elements in the newly allocated memory block
 *  \param elemSize - the size (in bytes) of each element in the newly allocated memory block
 * 
 *  \return
 *  ALLOC_SUCCESS - the memory is successfully allocated
 *    ALLOC_FAILURE - the memory allocation was unsuccessful
 */
__device__ int safeDeviceCalloc(void **pp, int elemCount, int elemSize);

/* Free a device memory block from a kernel, allocated by 'safeMalloc', 'safeCalloc' or 'safeRealloc'
 * The program will check if the pointer points to allocated memory
 * 
 *  \param pp - a pointer to the pointer to the allocated memory that needs to be freed
 *  \param elemCount - the number of elements in the newly allocated memory block
 *  \param elemSize - the size (in bytes) of each element in the newly allocated memory block
 * 
 *  \return
 *  FREE_SUCCESS if the memory is successfully freed
 *    FREE_FAILURE if *pp points to NULL
 */
__device__ int safeDeviceFree(void **pp);

#endif	/* SAFEMEM_CUH */

