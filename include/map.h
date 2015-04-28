/* 
 * File:   map.h
 * Author: rik
 *
 * Created on April 27, 2015, 7:33 PM
 */

#ifndef MAP_H
#define	MAP_H

#ifdef	__cplusplus
extern "C" {
#endif

    typedef struct _Map {
        int length;
        double *A;
        int *x;
        int *dx;
        int *y;
        int *dy;
        int *delta;
        int *phi;
    } *Map;

    void mallocMap(Map m, int p);
    void freeMap(Map m);
    
#ifdef	__cplusplus
}
#endif

#endif	/* MAP_H */

