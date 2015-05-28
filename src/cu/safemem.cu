/* 
 * File:   safemem.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:28 PM
 */
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../include/safemem.h"

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
    if (cudaMalloc(&p, elemCount * elemSize) != cudaSuccess) {
        fprintf(stderr, "The required space could not be allocated\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", elemCount, elemSize, elemCount * elemSize);
        return ALLOC_FAILURE;
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
        fprintf(stderr, " New Element count: %d\n", newElemCount);
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
        fprintf(stderr, "Tried to free a pointer to NULL\n");
        return FREE_FAILURE;
    }
    if (cudaFree(*pp) != cudaSuccess) {
        fprintf(stderr, "Memory is already freed\n");
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
        fprintf(stderr, "The elements could not be set to %d\n", value);
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
        fprintf(stderr, "The data could not be copied \n Element count: %d\n", elemCount);
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