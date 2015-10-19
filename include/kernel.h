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


	void sumArrayHelper(double *nums, int length, int interval);

	double sumArray(double *nums, int length);

	void calcData(DataArray dataArray, int iteration, Map map, double *newValue);

	void kernel(DataArray dataArray, Map x, Map dx, Map y, Map dy, Map delta, Map phi, int particleCount, int iterationCount);



#ifdef	__cplusplus
}
#endif

#endif	/* KERNEL_H */

