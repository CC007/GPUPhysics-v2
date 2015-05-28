/* 
 * File:   safemem.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:28 PM
 */
#include <stdio.h>
#include <stdlib.h>
#include "../../include/safemem.h"

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
int safeMalloc(void **pp, int elemCount, int elemSize) {
    void *p;

    p = malloc(elemCount * elemSize);
    if (p == NULL) {
        fprintf(stderr, "The required space could not be allocated\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", elemCount, elemSize, elemCount * elemSize);
        return ALLOC_FAILURE;
    }
    *pp = p;
    return ALLOC_SUCCESS;
}

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
int safeCalloc(void **pp, int elemCount, int elemSize) {
    void *p;

    p = calloc(elemCount, elemSize);
    if (p == NULL) {
        fprintf(stderr, "The required space could not be allocated\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", elemCount, elemSize, elemCount * elemSize);
        return ALLOC_FAILURE;
    }
    *pp = p;
    return ALLOC_SUCCESS;
}

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
int safeRealloc(void **pp, int elemCount, int elemSize) {
    void *p;

    p = realloc(*pp, elemCount * elemSize);
    if (p == NULL) {
        fprintf(stderr, "The required space could not be allocated\n Element count: %d\n Element size: %d\n Total size: %d bytes\n", elemCount, elemSize, elemCount * elemSize);
        return ALLOC_FAILURE;
    }
    *pp = p;
    return ALLOC_SUCCESS;
}

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
int safeFree(void **pp) {
    if (*pp != NULL) {
        free(*pp);
        *pp = NULL;
        return FREE_SUCCESS;
    }
    fprintf(stderr, "Tried to free a pointer to NULL\n");
    return FREE_FAILURE;
}