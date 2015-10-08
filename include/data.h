/* 
 * File:   data.h
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on April 27, 2015, 7:54 PM
 */

#ifndef DATA_H
#define	DATA_H

#ifdef	__cplusplus
extern "C" {
#endif

	typedef struct _DataArray {
		int length;
		double *x;
		double *dx;
		double *y;
		double *dy;
		double *delta;
		double *phi;
	} *DataArray;

	void mallocDataArray(DataArray *dataArray, int iterationCount, int particleCount);
	void freeDataArray(DataArray *dataArray, int particleCount);
	void cudaMallocDataArray(DataArray *dataArray, int iterationCount, int particleCount);
	void cudaFreeDataArray(DataArray *dataArray, int particleCount);

#ifdef	__cplusplus
}
#endif

#endif	/* DATA_H */

