/* 
 * File:   kernel.h
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on Oktober 4, 2015, 4:38 PM
 */

#ifndef KERNEL_H
#define	KERNEL_H

#ifdef	__cplusplus
extern "C" {
#endif


void sumArrayHelper(double *nums, int length, int interval) ;

double sumArray(double *nums, int length);

__device__ void cudaSumArrayHelper(double *nums, int length, int interval);

__device__ double cudaSumArray(double *nums, int length);



#ifdef	__cplusplus
}
#endif

#endif	/* KERNEL_H */

