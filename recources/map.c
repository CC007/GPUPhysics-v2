#include "map.h"

void mallocMap(Map m, int p) {
    m = malloc(sizeof (struct _Map));
    m->length = p;
    if (p > 0) {
        m->A = calloc(p, sizeof (double));
        m->x = calloc(p, sizeof (int));
        m->dx = calloc(p, sizeof (int));
        m->y = calloc(p, sizeof (int));
        m->dy = calloc(p, sizeof (int));
        m->delta = calloc(p, sizeof (int));
        m->phi = calloc(p, sizeof (int));
    }
}

void freeMap(Map m) {
    if (m->length > 0) {
        free(m->A);
        free(m->x);
        free(m->dx);
        free(m->y);
        free(m->dy);
        free(m->delta);
        free(m->phi);
    }
    free(m);
}