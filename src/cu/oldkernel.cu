#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define ITER 4000
#define MIN(x, y) (x<y?x:y)

void scanFile(FILE* fp, int *size) {
    char* line = (char*) malloc(200 * sizeof (char));
    line = fgets(line, 200, fp);
    if (strncmp(line, "     I  COEFFICIENT            ORDER EXPONENTS", 46) != 0) {
        if (strncmp(line, "     ALL COMPONENTS ZERO ", 25) != 0) {
            exit(EXIT_FAILURE);
        } else {
            *size = 1;
        }
    } else {
        for ((*size) = 0; !strstr((line = fgets(line, 200, fp)), "------"); (*size)++);
    }
    free(line);
}

void readMap(FILE *fp, Map *m, int nr) {
    char* line = (char*) malloc(200 * sizeof (char));
    line = fgets(line, 200, fp);
    int dum1, dum2;
    if (strncmp(line, "     I  COEFFICIENT            ORDER EXPONENTS", 46) != 0) {
        if (strncmp(line, "     ALL COMPONENTS ZERO ", 25) != 0) {
            exit(EXIT_FAILURE);
        } else {
            (*m).A[0] = 1.0;
            (*m).x[0] = nr == 0 ? 1 : 0;
            (*m).dx[0] = nr == 1 ? 1 : 0;
            (*m).y[0] = nr == 2 ? 1 : 0;
            (*m).dy[0] = nr == 3 ? 1 : 0;
            (*m).delta[0] = nr == 4 ? 1 : 0;
            (*m).phi[0] = nr == 5 ? 1 : 0;
        }
    }
    for (int i = 0; !strstr((line = fgets(line, 200, fp)), "------"); i++) {
        //TODO read chars ipv ints
        sscanf(line, "%d %lf %d %d %d %d %d %d %d",
                &dum1,
                &((*m).A[i]),
                &dum2,
                &((*m).x[i]),
                &((*m).dx[i]),
                &((*m).y[i]),
                &((*m).dy[i]),
                &((*m).delta[i]),
                &((*m).phi[i])
                );
    }
    free(line);
}

void readProperties(FILE *fp, Properties *v) {
    char* line = (char*) malloc(200 * sizeof (char));
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store muon mass\n");
    if (sscanf(line, "Muon Mass =   %lf MeV/c^2", &((*v).mass)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store muon momentum\n");
    if (sscanf(line, "Muon Momentum =   %lf MeV/c", &((*v).momentum)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store muon kin energy\n");
    if (sscanf(line, "Muon Kinetic Energy =   %lf MeV", &((*v).kinEn)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store Muon gamma\n");
    if (sscanf(line, "Muon gamma =   %lf", &((*v).gamma)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store Muon beta\n");
    if (sscanf(line, "Muon beta =  %lf", &((*v).beta)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store Muon Anomaly G\n");
    if (sscanf(line, "Muon Anomaly G =  %lf", &((*v).mAnomalyG)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    fprintf(stderr, "check and store Muon Spin Tune G.gamma\n");
    if (sscanf(line, "Muon Spin Tune G.gamma =  %lf", &((*v).spinTuneGgamma)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    if (sscanf(line, " L    %lf", &((*v).lRefOrbit)) != 1) exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    if (line[1] != 'P') exit(EXIT_FAILURE);
    line = fgets(line, 200, fp);
    if (line[1] != 'A') exit(EXIT_FAILURE);
    free(line);
}

void scanInputData(char *fileName, int *count) {
    char* line = (char*) malloc(200 * sizeof (char));
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error while opening the coefficients file: %s in ;", fileName);
        getchar();
        exit(EXIT_FAILURE);
    }
    for ((*count) = 0; fgets(line, 200, fp) != NULL; (*count)++) {
        if (strncmp(line, "\n", 1) == 0 || strncmp(line, "\0", 1) == 0) {
            (*count)--;
        }
    }
    if (!feof(fp)) {
        fprintf(stderr, "Something was wrong with the coefficient file!");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    free(line);
}

void getInputData(DataArray *c) {
    fprintf(stderr, "Begin values of the 6 dimentions: ");

    scanf("%lf %lf %lf %lf %lf %lf",
            &((*c).x[0]),
            &((*c).dx[0]),
            &((*c).y[0]),
            &((*c).dy[0]),
            &((*c).delta[0]),
            &((*c).phi[0])
            );
}

void readInputData(DataArray **c, char *fileName, int count) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error while opening the coefficients file: %s\n", fileName);
        getchar();
        exit(EXIT_FAILURE);
    }
    int i;
    for (i = 0; i < count; i++) {
        fscanf(fp, "%lf %lf %lf %lf %lf %lf",
                &((*c)[i].x[0]),
                &((*c)[i].dx[0]),
                &((*c)[i].y[0]),
                &((*c)[i].dy[0]),
                &((*c)[i].delta[0]),
                &((*c)[i].phi[0])
                );
    }
    fclose(fp);
}

void calcData(DataArray *c, int idx, Map *m, double *newValue) {
    double *nums = (double*) calloc((*m).length, sizeof (double));

    for (int i = 0; i < (*m).length; i++) {
        nums[i] = (*m).A[i] * pow((*c).x[idx], (*m).x[i])
                * pow((*c).dx[idx], (*m).dx[i])
                * pow((*c).y[idx], (*m).y[i])
                * pow((*c).dy[idx], (*m).dy[i])
                * pow((*c).delta[idx], (*m).delta[i])
                * pow((*c).phi[idx], (*m).phi[i]);
    }

    *newValue = sumArray(nums, (*m).length);
    free(nums);
}

__device__ void cudaCalcData(DataArray *c, int idx, Map *m, double *newValue) {
    double *nums = (double*) malloc((*m).length * sizeof (double));
    memset(nums, 0, (*m).length * sizeof (double));
    for (int i = 0; i < (*m).length; i++) {
        nums[i] = (*m).A[i] * pow((*c).x[idx], (*m).x[i])
                * pow((*c).dx[idx], (*m).dx[i])
                * pow((*c).y[idx], (*m).y[i])
                * pow((*c).dy[idx], (*m).dy[i])
                * pow((*c).delta[idx], (*m).delta[i])
                * pow((*c).phi[idx], (*m).phi[i]);
    }

    *newValue = cudaSumArray(nums, (*m).length);
    free(nums);
}

void kernel(DataArray *c, Map *x, Map *dx, Map *y, Map *dy, Map *delta, Map *phi, int particleCount, int iter) {
    for (int n = 0; n < particleCount; n++) {
        for (int i = 0; i < iter - 1; i++) {
            calcData(&(c[n]), i, x, &(c[n].x[i + 1]));
            calcData(&(c[n]), i, dx, &(c[n].dx[i + 1]));
            calcData(&(c[n]), i, y, &(c[n].y[i + 1]));
            calcData(&(c[n]), i, dy, &(c[n].dy[i + 1]));
            calcData(&(c[n]), i, delta, &(c[n].delta[i + 1]));
            calcData(&(c[n]), i, phi, &(c[n].phi[i + 1]));
        }
    }
}

__global__ void cudaKernel(DataArray *c, Map *x, Map *dx, Map *y, Map *dy, Map *delta, Map *phi, int particleCount, int iter) {
    int sizeX = gridDim.x;
    int idx = blockIdx.x;
    int sizeY = blockDim.x;
    int idy = threadIdx.x;
    for (int n = idx * sizeY + idy; n < particleCount; n += sizeX * sizeY) {
        for (int i = 0; i < iter - 1; i++) {
            cudaCalcData(&(c[n]), i, x, &(c[n].x[i + 1]));
            cudaCalcData(&(c[n]), i, dx, &(c[n].dx[i + 1]));
            cudaCalcData(&(c[n]), i, y, &(c[n].y[i + 1]));
            cudaCalcData(&(c[n]), i, dy, &(c[n].dy[i + 1]));
            cudaCalcData(&(c[n]), i, delta, &(c[n].delta[i + 1]));
            cudaCalcData(&(c[n]), i, phi, &(c[n].phi[i + 1]));
        }
    }
}

int main(int argc, char **argv) {
    char fileName[200] = "";
    char inputFileName[200] = "";
    char *outputFileName = (char*) malloc(200 * sizeof (char));
    outputFileName[0] = '\0';
    int separateFiles = 0, accelerate = 0;
    int xSize, dxSize, ySize, dySize, deltaSize, phiSize, argcCounter, particleCount = 1, iter = ITER;
    Map x, dx, y, dy, delta, phi;
    Map *dev_x, *dev_dx, *dev_y, *dev_dy, *dev_delta, *dev_phi;
    DataArray *c;
    DataArray *dev_c;
    Properties v;

    float h2dTime = 0, kTime = 0, d2hTime;
    cudaEvent_t start, stopH2D, stopK, stopD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&stopK);
    cudaEventCreate(&stopD2H);


    // Read the program arguments
    argcCounter = argc;
    while ((argcCounter > 1) && (argv[1][0] == '-')) {
        if (argv[1][2] == '=' && argv[1][3] != '\0') {
            switch (argv[1][1]) {
                case 'm':
                    sprintf(fileName, "%s", &argv[1][3]);
                    break;

                case 'c':
                    sprintf(inputFileName, "%s", &argv[1][3]);
                    break;

                case 'o':
                    sprintf(outputFileName, "%s", &argv[1][3]);
                    break;

                case 'i':
                    sscanf(&argv[1][3], "%d", &iter);
                    break;

                default:
                    fprintf(stderr, "Wrong Argument: %s\n", argv[1]);
                    getchar();
                    exit(EXIT_FAILURE);
            }
        } else {
            switch (argv[1][1]) {
                case 's':
                    separateFiles = 1;
                    break;
                case 'g':
                    if (!strstr(&argv[1][2], "pu") || argv[1][4] != '\0') {
                        fprintf(stderr, "Wrong Argument: %s\n", argv[1]);
                        getchar();
                        exit(EXIT_FAILURE);
                    } else {
                        accelerate = 1;
                    }
                    break;
                case '-':
                    if (!strstr(&argv[1][2], "help") || argv[1][6] != '\0') {
                        fprintf(stderr, "Wrong Argument: %s\n", argv[1]);
                        getchar();
                        exit(EXIT_FAILURE);
                    }
                case 'h':
                    if (strstr(&argv[1][2], "help") || argv[1][2] == '\0') {
                        printf("Calculates a certain amount of steps of a charged particle in a inhomogeneous\nmagnetic field.\n\n");
                        printf("<executable> (-h|--help) | <executable> [-m=<mapFileName>] [-c=<coeffFileName>]\n[-o=<outputFileName> [-s]] [-gpu] [-i=<nr>]\n\n");
                        printf("-h, --help\t\t Display help\n");
                        printf("-m=<mapFileName>\t Set the map file to be <mapFileName>. If not set, it\n\t\t\t will be asked for in the program itself.\n");
                        printf("-c=<coeffFileName>\t Set the coefficients file to be <coeffFileName>. If\n\t\t\t not set, it will be asked for in the program itself.\n\t\t\t Note that the coefficients file supports multiple\n\t\t\t particles, while if the program is run without this\n\t\t\t file, it supports only one particle.\n");
                        printf("-o=<outputFileName>\t Set the output file to be <outputFileName>. If not\n\t\t\t set, it will default to stdout\n");
                        printf("-s\t\t\t Choose if you want one output file or (if applicable)\n\t\t\t multiple output files. Note that this parameter can\n\t\t\t only be set if an output file is set.\n");
                        printf("-gpu\t\t\t Choose if you want to use GPU acceleration. You would\n\t\t\t need a NVIDIA videocard with compute capability 2.0\n\t\t\t or higher (Fermi microarchitecture).\n");
                        printf("-i=<nr>\t\t\t Set the number of iterations to <nr>. If not set, it\n\t\t\t will default to 4000.\n\n");
                        printf("Press Enter to continue...\n");
                        getchar();
                        exit(EXIT_SUCCESS);
                    } else {
                        fprintf(stderr, "Wrong Argument: %s\n", argv[1]);
                        getchar();
                        exit(EXIT_FAILURE);
                    }
                    break;

                default:
                    fprintf(stderr, "Wrong Argument: %s\n", argv[1]);
                    getchar();
                    exit(EXIT_FAILURE);
            }
        }
        ++argv;
        --argcCounter;
    }
    if (separateFiles == 1 && strncmp(outputFileName, "\0", 1) == 0) {
        fprintf(stderr, "-s shouldn't be used without setting an output file");
        exit(EXIT_FAILURE);
    }

    // if not set in argument, ask for file name of the map file
    if (strncmp(fileName, "\0", 1) == 0) {
        fprintf(stderr, "Filename of the map: ");
        scanf("%s", fileName);
    }

    // use the map file to gather the sizes of the 6 coefficients
    fprintf(stderr, "open file\n");
    FILE *scanFileP = fopen(fileName, "r");
    fprintf(stderr, "check if file is NULL\n");
    if (scanFileP == NULL) {
        fprintf(stderr, "Error while opening the map file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Get map sizes\n");
    char* line = (char*) malloc(200 * sizeof (char));
    do {
        line = fgets(line, 200, scanFileP);
    } while (!strstr(line, " A "));
    free(line);
    scanFile(scanFileP, &xSize);
    scanFile(scanFileP, &dxSize);
    scanFile(scanFileP, &ySize);
    scanFile(scanFileP, &dySize);
    scanFile(scanFileP, &deltaSize);
    scanFile(scanFileP, &phiSize);
    fclose(scanFileP);
    fprintf(stderr, "\nmap sizes: %d %d %d %d %d %d\n", xSize, dxSize, ySize, dySize, deltaSize, phiSize);

    // allocate memory for the map
    fprintf(stderr, "\nmap x\n");
    mallocMap(&x, xSize);
    cudaMallocMap(&dev_x, xSize);
    fprintf(stderr, "map dx\n");
    mallocMap(&dx, dxSize);
    cudaMallocMap(&dev_dx, dxSize);
    fprintf(stderr, "map y\n");
    mallocMap(&y, ySize);
    cudaMallocMap(&dev_y, ySize);
    fprintf(stderr, "map dy\n");
    mallocMap(&dy, dySize);
    cudaMallocMap(&dev_dy, dySize);
    fprintf(stderr, "map delta\n");
    mallocMap(&delta, deltaSize);
    cudaMallocMap(&dev_delta, deltaSize);
    fprintf(stderr, "map phi\n");
    mallocMap(&phi, phiSize);
    cudaMallocMap(&dev_phi, phiSize);

    // read some variables and the map lines from the map file
    fprintf(stderr, "open file\n");
    FILE *mapFileP = fopen(fileName, "r");
    fprintf(stderr, "check if file is NULL\n");
    if (mapFileP == NULL) {
        fprintf(stderr, "Error while opening the map file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "read properties\n");
    readProperties(mapFileP, &v);
    fprintf(stderr, "read x\n");
    readMap(mapFileP, &x, 0);
    fprintf(stderr, "read dx\n");
    readMap(mapFileP, &dx, 1);
    fprintf(stderr, "read y\n");
    readMap(mapFileP, &y, 2);
    fprintf(stderr, "read dy\n");
    readMap(mapFileP, &dy, 3);
    fprintf(stderr, "read delta\n");
    readMap(mapFileP, &delta, 4);
    fprintf(stderr, "read phi\n");
    readMap(mapFileP, &phi, 5);
    fclose(mapFileP);


    // allocate mempry for the coefficients

    // read the coefficients from user input
    if (strncmp(inputFileName, "\0", 1) == 0) {
        fprintf(stderr, "malloc data\n");
        mallocDataArray(&c, iter, particleCount);
        cudaMallocDataArray(&dev_c, iter, particleCount);
        fprintf(stderr, "read input\n");
        getInputData(c);
    } else {
        scanInputData(inputFileName, &particleCount);
        fprintf(stderr, "malloc data\n");
        mallocDataArray(&c, iter, particleCount);
        cudaMallocDataArray(&dev_c, iter, particleCount);
        fprintf(stderr, "read input\n");
        readInputData(&c, inputFileName, particleCount);
        fprintf(stderr, "Particle count: %d\n", particleCount);
    }

    // calculate the coefficients for 4000 iterations
    if (!accelerate) {
        //cpu
        kernel(c, &x, &dx, &y, &dy, &delta, &phi, particleCount, iter);
    } else {
        //gpu
        cudaEventRecord(start);
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, 0);
        int blocks = MIN(1048576, particleCount);
        fprintf(stderr, "Device name: %s\nUsed blocks: %d\n", deviceProperties.name, blocks);
        int dimBlocks = sqrt((float) blocks);
        int dimThreads = blocks / dimBlocks + (blocks % dimBlocks > 0);
        cudaMemcpyMap(dev_x, &x, cudaMemcpyHostToDevice);
        cudaMemcpyMap(dev_dx, &dx, cudaMemcpyHostToDevice);
        cudaMemcpyMap(dev_y, &y, cudaMemcpyHostToDevice);
        cudaMemcpyMap(dev_dy, &dy, cudaMemcpyHostToDevice);
        cudaMemcpyMap(dev_delta, &delta, cudaMemcpyHostToDevice);
        cudaMemcpyMap(dev_phi, &phi, cudaMemcpyHostToDevice);
        cudaMemcpyFirstDataArray(dev_c, c, particleCount);

        cudaEventRecord(stopH2D);
        
        cudaKernel<<<dimBlocks, dimThreads>>>(dev_c, dev_x, dev_dx, dev_y, dev_dy, dev_delta, dev_phi, particleCount, iter);
        
        cudaEventRecord(stopK);
    }
    for (int n = 0; n < particleCount; n++) {
        // if acceleration is on, copy input from device to host
        if (accelerate) {
            cudaMemcpyDataArray(&(c[n]), &(dev_c[n]), cudaMemcpyDeviceToHost);
        }
        // show or save
        FILE* outputFile;
        char fullOutputFileName[200] = "";
        if (!strncmp(outputFileName, "\0", 1) == 0) {
            if (separateFiles == 1) {
                sprintf(fullOutputFileName, "%s.part%09d", outputFileName, n + 1); //bug if path is used to file
            } else {
                sprintf(fullOutputFileName, "%s", outputFileName);
            }
            if (n == 0) {
                outputFile = fopen(fullOutputFileName, "w");
            } else {
                if (separateFiles == 1) {
                    outputFile = fopen(fullOutputFileName, "w");
                } else {
                    outputFile = fopen(fullOutputFileName, "a");
                }
            }
            if (outputFile == NULL) {
                fprintf(stderr, "Error while opening the output file: %s\n", fullOutputFileName);
                exit(EXIT_FAILURE);
            }
        } else {
            outputFile = stdout;
        }
        if (accelerate) {
            cudaMemcpyDataArray(&(c[n]), &(dev_c[n]), cudaMemcpyDeviceToHost);
        }
        for (int i = 0; i < iter; i++) {
            fprintf(outputFile, "%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n", c[n].x[i], c[n].dx[i], c[n].y[i], c[n].dy[i], c[n].delta[i], c[n].phi[i]);
        }
        fprintf(outputFile, "\n");
        if (!strncmp(outputFileName, "\0", 1) == 0) {
            fclose(outputFile);
        }
    }
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    cudaEventElapsedTime(&h2dTime, start, stopH2D);
    cudaEventElapsedTime(&kTime, stopH2D, stopK);
    cudaEventElapsedTime(&d2hTime, stopK, stopD2H);
    
    
    fprintf(stderr, "Elapsed time:\n\n H2D:\t%f\nKernel:\t%f\nD2H:\t%f", h2dTime, kTime, d2hTime);

    // clean up the heap and tell that the computation is finished
    freeMap(&x);
    freeMap(&dx);
    freeMap(&y);
    freeMap(&dy);
    freeMap(&delta);
    freeMap(&phi);
    cudaFreeMap(&dev_x);
    cudaFreeMap(&dev_dx);
    cudaFreeMap(&dev_y);
    cudaFreeMap(&dev_dy);
    cudaFreeMap(&dev_delta);
    cudaFreeMap(&dev_phi);
    freeDataArray(&c, particleCount);
    cudaFreeDataArray(&dev_c, particleCount);
    free(outputFileName);
    fprintf(stderr, "Output is created. Press Enter to continue...\n");
    getchar();
    if (strncmp(inputFileName, "\0", 1) == 0) {
        getchar();
    }
    return 0;
}
