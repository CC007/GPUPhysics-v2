/* 
 * File:   safemem.h
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:49 PM
 */

#ifndef SAFEMEM_H
#define	SAFEMEM_H

#include "safememstates.h"

#ifdef	__cplusplus
extern "C" {
#endif

	/* Allocate elemCount * elemSize bytes of memory
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
	int safeMalloc(void **returnPointer, int elementCount, int elementSize);

	/* Allocate elemCount * elemSize bytes of memory, all elements initialized to 0
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
	int safeCalloc(void **pp, int elemCount, int elemSize);

	/* Reallocate elemCount * elemSize bytes of memory
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
	int safeRealloc(void **pp, int elemCount, int elemSize);

	/* Free a memory block, allocated by 'safeMalloc', 'safeCalloc' or 'safeRealloc'
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
	int safeFree(void **pp);



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
	int safeCudaMalloc(void **pp, int elemCount, int elemSize);

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
	int safeCudaCalloc(void **pp, int elemCount, int elemSize);

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
	int safeCudaRealloc(void **pp, int oldElemCount, int newElemCount, int elemSize);

	/* Free a device memory block, allocated by 'safeCudaMalloc', 'safeCudaCalloc' or 'safeCudaRealloc'
	 * The program will check if the pointer points to allocated memory
	 * 
	 *  \param pp - a pointer to the pointer to the allocated memory that needs to be freed
	 * 
	 *  \return
	 *  FREE_SUCCESS if the memory is successfully freed
	 *    FREE_FAILURE if *pp points to NULL
	 */
	int safeCudaFree(void **pp);

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
	int safeCudaMemset(void *p, int value, int elemCount, int elemSize);

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
	int safeCudaMemcpyHtD(void *destination, const void *source, int elemCount, int elemSize);

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
	int safeCudaMemcpyDtH(void *destination, const void *source, int elemCount, int elemSize);

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
	int safeCudaMemcpyDtD(void *destination, const void *source, int elemCount, int elemSize);
#ifdef	__cplusplus
}
#endif

#endif	/* SAFEMEM_H */

