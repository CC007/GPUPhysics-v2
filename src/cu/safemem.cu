/* 
 * File:   safemem.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:28 PM
 */
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/extendedio.h"
#include "../../include/safemem.h"
#include "../../include/safemem.cuh"

/* Allocate elemCount * elemSize bytes of device memory
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
int safeCudaMalloc(void **pp, int elemCount, int elemSize) {
    void *p;
	for(int i = 0; cudaMalloc(&p, elemCount * elemSize) != cudaSuccess; i++){
		if(i > 100){
			wprintf("The required space could not be allocated after %d attempts.\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", i, elemCount, elemSize, elemCount * elemSize);
			return ALLOC_FAILURE;
		}
	}
    *pp = p;
    return ALLOC_SUCCESS;
}

/* Allocate elemCount * elemSize bytes of device memory, all elements initialized to 0
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
int safeCudaCalloc(void **pp, int elemCount, int elemSize) {
    void *p;
    if (safeCudaMalloc(&p, elemCount, elemSize)) {
        return ALLOC_FAILURE;
    }
    if (safeCudaMemset(p, 0, elemCount, elemSize)) {
        return ALLOC_FAILURE;
    }
    *pp = p;
    return ALLOC_SUCCESS;
}

/* Reallocate elemCount * elemSize bytes of device memory
 * The program will check if the memory was successfully allocated
 * 
 *  \param pp - a pointer to the pointer that afterwards points to the allocated memory
 *  \param oldElemCount - the number of elements in the old memory block
 *  \param newElemCount - the number of elements in the newly allocated memory block
 *  \param elemSize - the size (in bytes) of each element in the newly allocated memory block
 * 
 *  \return
 *  ALLOC_SUCCESS - the memory is successfully allocated
 *    ALLOC_FAILURE - the memory allocation was unsuccessful
 */
int safeCudaRealloc(void **pp, int oldElemCount, int newElemCount, int elemSize) {
    void *p;

    if (safeCudaMalloc(&p, newElemCount, elemSize)) {
        return ALLOC_FAILURE;
    }
    if (safeCudaMemcpyDtD(p, *pp, oldElemCount, elemSize)) {
        wprintf(" New Element count: %d\n", newElemCount);
        return ALLOC_FAILURE;
    }
    safeCudaFree(pp); // no problem if it fails, but will probably mean that there is a mem leak
    *pp = p;
    return ALLOC_SUCCESS;
}

/* Free a device memory block, allocated by 'safeCudaMalloc', 'safeCudaCalloc' or 'safeCudaRealloc'
 * The program will check if the pointer points to allocated memory
 * 
 *  \param pp - a pointer to the pointer to the allocated memory that needs to be freed
 * 
 *  \return
 *  FREE_SUCCESS if the memory is successfully freed
 *    FREE_FAILURE if *pp points to NULL
 */
int safeCudaFree(void **pp) {
    if (*pp == NULL) {
        wprintf("Tried to free a pointer to NULL\n");
        return FREE_FAILURE;
    }
    if (cudaFree(*pp) != cudaSuccess) {
        wprintf("Memory is already freed\n");
        return FREE_FAILURE;
    }
    *pp = NULL;
    return FREE_SUCCESS;
}

/* Set elemCount * elemSize bytes of device memory to a specified value
 * The program will check if the values have successfully been set
 * 
 *  \param p - the pointer to the memory that needs to be set to a value
 *  \param value - the value that the memory should be set to
 *  \param elemCount - the number of elements in the memory block
 *  \param elemSize - the size (in bytes) of each element in the memory block
 * 
 *  \return
 *  MEMSET_SUCCESS - the memory is successfully set to the specified value
 *    MEMSET_FAILURE - setting the memory to the specified value was unsuccessful
 */
int safeCudaMemset(void *p, int value, int elemCount, int elemSize) {
    if (cudaMemset(p, value, elemCount * elemSize) != cudaSuccess) {
        wprintf("The elements could not be set to %d\n", value);
        return MEMSET_FAILURE;
    }
    return MEMSET_SUCCESS;
}

/* Copies data between host and device
 * The program will check if the data was successfully copied
 * 
 *  \param destination - a pointer to the destination memory
 *  \param source - a pointer to the source memory
 *  \param elemCount - the number of elements to be copied
 *  \param elemSize - the size (in bytes) of each element in the memory block
 *  \param kind - type of transfer (device -> host / device -> device / host -> host / host -> device)
 * 
 *  \return
 *  MEMCPY_SUCCESS - the memory is successfully copied
 *    MEMCPY_FAILURE - copying the memory was unsuccessful
 */
int safeCudaMemcpy(void *destination, const void *source, int elemCount, int elemSize, cudaMemcpyKind kind) {
    if (cudaMemcpy(destination, source, elemCount * elemSize, kind) != cudaSuccess) {
        wprintf("The data could not be copied \n");
		iprintf("Element count: %d\n", elemCount);
        return MEMCPY_FAILURE;
    }
    return MEMCPY_SUCCESS;
}

/* Copies data from host to device
 * The program will check if the data was successfully copied
 * 
 *  \param destination - a pointer to the destination memory
 *  \param source - a pointer to the source memory
 *  \param elemCount - the number of elements to be copied
 *  \param elemSize - the size (in bytes) of each element in the memory block
 * 
 *  \return
 *  MEMCPY_SUCCESS - the memory is successfully copied
 *    MEMCPY_FAILURE - copying the memory was unsuccessful
 */
int safeCudaMemcpyHtD(void *destination, const void *source, int elemCount, int elemSize) {
    if (safeCudaMemcpy(destination, source, elemCount, elemSize, cudaMemcpyHostToDevice)) {
        return MEMCPY_FAILURE;
    }
    return MEMCPY_SUCCESS;
}

/* Copies data from device to host
 * The program will check if the data was successfully copied
 * 
 *  \param destination - a pointer to the destination memory
 *  \param source - a pointer to the source memory
 *  \param elemCount - the number of elements to be copied
 *  \param elemSize - the size (in bytes) of each element in the memory block
 * 
 *  \return
 *  MEMCPY_SUCCESS - the memory is successfully copied
 *    MEMCPY_FAILURE - copying the memory was unsuccessful
 */
int safeCudaMemcpyDtH(void *destination, const void *source, int elemCount, int elemSize) {
    if (safeCudaMemcpy(destination, source, elemCount, elemSize, cudaMemcpyDeviceToHost)) {
        return MEMCPY_FAILURE;
    }
    return MEMCPY_SUCCESS;
}

/* Copies data from device to device
 * The program will check if the data was successfully copied
 * 
 *  \param destination - a pointer to the destination memory
 *  \param source - a pointer to the source memory
 *  \param elemCount - the number of elements to be copied
 *  \param elemSize - the size (in bytes) of each element in the memory block
 * 
 *  \return
 *  MEMCPY_SUCCESS - the memory is successfully copied
 *    MEMCPY_FAILURE - copying the memory was unsuccessful
 */
int safeCudaMemcpyDtD(void *destination, const void *source, int elemCount, int elemSize) {
    if (safeCudaMemcpy(destination, source, elemCount, elemSize, cudaMemcpyDeviceToDevice)) {
        return MEMCPY_FAILURE;
    }
    return MEMCPY_SUCCESS;
}

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
__device__ int safeDeviceMalloc(void **pp, int elemCount, int elemSize) {
    void *p = NULL;

	for(int i = 0; p == NULL; i++){
		if(i > 100){
			printf("The required space could not be allocated after %d attempts.\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", i, elemCount, elemSize, elemCount * elemSize);
			return ALLOC_FAILURE;
		}
		p = malloc(elemCount * elemSize);
	}
    *pp = p;
    return ALLOC_SUCCESS;
}

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
__device__ int safeDeviceCalloc(void **pp, int elemCount, int elemSize) {
    void *p;
    if (safeDeviceMalloc(&p, elemCount, elemSize)) {
        return ALLOC_FAILURE;
    }
	memset(p, 0, elemCount * elemSize); // should always work if allocation succeeded
    *pp = p;
    return ALLOC_SUCCESS;
}

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
__device__ int safeDeviceFree(void **pp) {
    if (*pp != NULL) {
        free(*pp);
        *pp = NULL;
        return FREE_SUCCESS;
    }
    printf("Tried to free a pointer to NULL\n");
    return FREE_FAILURE;
}