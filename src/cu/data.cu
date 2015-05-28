/* 
 * File:   data.cu
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on May 27, 2015, 9:03 PM
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../include/data.h"
#include "../../include/safemem.h"

void cudaMallocData(Data *c, int iter, int p) {
    if (iter > 0) {
        int i;
        cudaMalloc((void**) c, p * sizeof (Data));
        for (i = 0; i < p; i++) {
            Data helper_d;
            helper_d.length = iter;
            cudaMalloc((void**) &(helper_d.x), iter * sizeof (double));
            cudaMalloc((void**) &(helper_d.dx), iter * sizeof (double));
            cudaMalloc((void**) &(helper_d.y), iter * sizeof (double));
            cudaMalloc((void**) &(helper_d.dy), iter * sizeof (double));
            cudaMalloc((void**) &(helper_d.delta), iter * sizeof (double));
            cudaMalloc((void**) &(helper_d.phi), iter * sizeof (double));
            cudaMemcpy(&((*c)[i]), &helper_d, sizeof (Data), cudaMemcpyHostToDevice);
        }
    }
}

void cudaMemcpyMap(Map *dst_m, Map *src_m, cudaMemcpyKind kind) {
    Map helper_m;
    if (kind == cudaMemcpyDeviceToHost) {
        cudaMemcpy(&helper_m, src_m, sizeof (Map), cudaMemcpyDeviceToHost);
        cudaMemcpy(dst_m->A, helper_m.A, helper_m.length * sizeof (double), kind);
        cudaMemcpy(dst_m->x, helper_m.x, helper_m.length * sizeof (int), kind);
        cudaMemcpy(dst_m->dx, helper_m.dx, helper_m.length * sizeof (int), kind);
        cudaMemcpy(dst_m->y, helper_m.y, helper_m.length * sizeof (int), kind);
        cudaMemcpy(dst_m->dy, helper_m.dy, helper_m.length * sizeof (int), kind);
        cudaMemcpy(dst_m->delta, helper_m.delta, helper_m.length * sizeof (int), kind);
        cudaMemcpy(dst_m->phi, helper_m.phi, helper_m.length * sizeof (int), kind);
    } else if (kind == cudaMemcpyHostToDevice) {
        cudaMemcpy(&helper_m, dst_m, sizeof (Map), cudaMemcpyDeviceToHost);
        cudaMemcpy(helper_m.A, src_m->A, helper_m.length * sizeof (double), kind);
        cudaMemcpy(helper_m.x, src_m->x, helper_m.length * sizeof (int), kind);
        cudaMemcpy(helper_m.dx, src_m->dx, helper_m.length * sizeof (int), kind);
        cudaMemcpy(helper_m.y, src_m->y, helper_m.length * sizeof (int), kind);
        cudaMemcpy(helper_m.dy, src_m->dy, helper_m.length * sizeof (int), kind);
        cudaMemcpy(helper_m.delta, src_m->delta, helper_m.length * sizeof (int), kind);
        cudaMemcpy(helper_m.phi, src_m->phi, helper_m.length * sizeof (int), kind);
    } else {
        fprintf(stderr, "DeviceToDevice is not yet supported for maps!\n");
        getchar();
        exit(EXIT_FAILURE);
    }

}