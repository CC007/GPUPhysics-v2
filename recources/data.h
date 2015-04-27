/* 
 * File:   input.h
 * Author: rik
 *
 * Created on April 27, 2015, 7:54 PM
 */

#ifndef INPUT_H
#define	INPUT_H

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct Data {
    int length;
    double *x;
    double *dx;
    double *y;
    double *dy;
    double *delta;
    double *phi;
} *Data;





#ifdef	__cplusplus
}
#endif

#endif	/* INPUT_H */

