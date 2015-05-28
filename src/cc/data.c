/* 
 * File:   data.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on April 27, 2015, 7:54 PM
 */

#include "../../include/data.h"
#include "../../include/safemem.h"

void mallocData(Data *c, int iter, int p) {
    if (iter > 0) {
        int i;
        c = malloc(p * sizeof (Data));
        for (i = 0; i < p; i++) {
            c[i]->length = iter;
            c[i]->x = calloc(iter, sizeof (double));
            c[i]->dx = calloc(iter, sizeof (double));
            c[i]->y = calloc(iter, sizeof (double));
            c[i]->dy = calloc(iter, sizeof (double));
            c[i]->delta = calloc(iter, sizeof (double));
            c[i]->phi = calloc(iter, sizeof (double));
        }
    }
}

void freeData(Data *c, int p) {
    if (c[0]->length > 0) {
        int i;
        for (i = 0; i < p; i++) {
            free(c[i]->x);
            free(c[i]->dx);
            free(c[i]->y);
            free(c[i]->dy);
            free(c[i]->delta);
            free(c[i]->phi);
        }
        free(c);
    }
}
