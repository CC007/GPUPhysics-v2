#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../include/data.h"
#include "../../include/data.cuh"
#include "../../include/map.h"
#include "../../include/map.cuh"
#include "../../include/spinmap.h"
#include "../../include/spinmap.cuh"
#include "../../include/properties.h"
#include "../../include/kernel.h"
#include "../../include/kernel.cuh"
#include "../../include/extendedio.h"
#include "../../include/safemem.h"

#define ITER 4000
#define MIN(x, y) (x<y?x:y)

void scanMap(FILE* fp, int *size) {
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

void readMap(FILE *fp, Map map, int nr) {
	char* line = (char*) malloc(200 * sizeof (char));
	line = fgets(line, 200, fp);
	int dum1, dum2;
	if (strncmp(line, "     I  COEFFICIENT            ORDER EXPONENTS", 46) != 0) {
		if (strncmp(line, "     ALL COMPONENTS ZERO ", 25) != 0) {
			eprintf("Corrupt line in file: %s", line);
		} else {
			map->A[0] = 1.0;
			map->x[0] = nr == 0 ? 1 : 0;
			map->dx[0] = nr == 1 ? 1 : 0;
			map->y[0] = nr == 2 ? 1 : 0;
			map->dy[0] = nr == 3 ? 1 : 0;
			map->delta[0] = nr == 4 ? 1 : 0;
			map->phi[0] = nr == 5 ? 1 : 0;
		}
	}
	for (int i = 0; !strstr((line = fgets(line, 200, fp)), "------"); i++) {
		//TODO read chars ipv ints
		sscanf(line, "%d %lf %d %d %d %d %d %d %d",
				&dum1,
				&(map->A[i]),
				&dum2,
				&(map->x[i]),
				&(map->dx[i]),
				&(map->y[i]),
				&(map->dy[i]),
				&(map->delta[i]),
				&(map->phi[i])
				);
	}
	free(line);
}

void scanSpinMap(FILE* fp, int *size) {
	char* line = (char*) malloc(200 * sizeof (char));
	line = fgets(line, 200, fp);
	if (strncmp(line, "     ALL COMPONENTS ZERO ", 25) != 0) {
		eprintf("Corrupt line in file: \n found: \"%s\"\n expected: \"     ALL COMPONENTS ZERO \"", line);
	}
	if (!strstr((line = fgets(line, 200, fp)), "------")) {
		eprintf("Corrupt line in file: %s", line);
	}
	line = fgets(line, 200, fp);
	if (strncmp(line, "     ALL COMPONENTS ZERO ", 25) != 0) {
		eprintf("Corrupt line in file: \n found: \"%s\"\n expected: \"     ALL COMPONENTS ZERO \"", line);
	}
	if (!strstr((line = fgets(line, 200, fp)), "------")) {
		eprintf("Corrupt line in file: %s", line);
	}
	for (int i = 0; i < 3; i++) {
		for (size[i] = 0; !strstr((line = fgets(line, 200, fp)), "------"); size[i]++);
	}
	free(line);
}

void readSpinMap(FILE *fp, InnerSpinMap innerSpinMap) {
	char* line = (char*) malloc(200 * sizeof (char));
	int exponents;
	int x, dx, y, dy, delta, phi;
	for (int i = 0; !strstr((line = fgets(line, 200, fp)), "------"); i++) {
		//TODO read chars ipv ints
		sscanf(line, "%lf %lf %lf %d",
				&(innerSpinMap->x->A[i]),
				&(innerSpinMap->y->A[i]),
				&(innerSpinMap->z->A[i]),
				&(exponents)
				);
		phi = exponents % 10;
		exponents /= 10;
		delta = exponents % 10;
		exponents /= 10;
		dy = exponents % 10;
		exponents /= 10;
		y = exponents % 10;
		exponents /= 10;
		dx = exponents % 10;
		exponents /= 10;
		x = exponents % 10;
		innerSpinMap->x->x[i] = x;
		innerSpinMap->x->dx[i] = dx;
		innerSpinMap->x->y[i] = y;
		innerSpinMap->x->dy[i] = dy;
		innerSpinMap->x->delta[i] = delta;
		innerSpinMap->x->phi[i] = phi;

		innerSpinMap->y->x[i] = x;
		innerSpinMap->y->dx[i] = dx;
		innerSpinMap->y->y[i] = y;
		innerSpinMap->y->dy[i] = dy;
		innerSpinMap->y->delta[i] = delta;
		innerSpinMap->y->phi[i] = phi;

		innerSpinMap->z->x[i] = x;
		innerSpinMap->z->dx[i] = dx;
		innerSpinMap->z->y[i] = y;
		innerSpinMap->z->dy[i] = dy;
		innerSpinMap->z->delta[i] = delta;
		innerSpinMap->z->phi[i] = phi;
	}
	free(line);
}

void readProperties(FILE *fp, Properties *properties) {
	char* line = (char*) malloc(200 * sizeof (char));
	line = fgets(line, 200, fp);
	iprintf("check and store muon mass\n");
	if (sscanf(line, "Muon Mass =   %lf MeV/c^2", &(properties->mass)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store muon momentum\n");
	if (sscanf(line, "Muon Momentum =   %lf MeV/c", &(properties->momentum)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store muon kin energy\n");
	if (sscanf(line, "Muon Kinetic Energy =   %lf MeV", &(properties->kinEn)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store Muon gamma\n");
	if (sscanf(line, "Muon gamma =   %lf", &(properties->gamma)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store Muon beta\n");
	if (sscanf(line, "Muon beta =  %lf", &(properties->beta)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store Muon Anomaly G\n");
	if (sscanf(line, "Muon Anomaly G =  %lf", &(properties->mAnomalyG)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	iprintf("check and store Muon Spin Tune G.gamma\n");
	if (sscanf(line, "Muon Spin Tune G.gamma =  %lf", &(properties->spinTuneGgamma)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	if (sscanf(line, " L    %lf", &(properties->lRefOrbit)) != 1) exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	if (line[1] != 'P') exit(EXIT_FAILURE);
	line = fgets(line, 200, fp);
	if (line[1] != 'A') exit(EXIT_FAILURE);
	free(line);
}

void scanDataArray(char *fileName, int *count) {
	char* line = (char*) malloc(200 * sizeof (char));
	FILE *fp = fopen(fileName, "r");
	if (fp == NULL) {
		eprintf("Error while opening the data array file: %s in ;", fileName);
	}
	for ((*count) = 0; fgets(line, 200, fp) != NULL; (*count)++) {
		if (strncmp(line, "\n", 1) == 0 || strncmp(line, "\0", 1) == 0) {
			(*count)--;
		}
	}
	if (!feof(fp)) {
		eprintf("Something was wrong with the data array file!");
	}
	fclose(fp);
	free(line);
}

void getDataArray(DataArray dataArray) {
	iprintf("Begin values of the 6 dimentions: ");

	scanf("%lf %lf %lf %lf %lf %lf",
			&(dataArray->x[0]),
			&(dataArray->dx[0]),
			&(dataArray->y[0]),
			&(dataArray->dy[0]),
			&(dataArray->delta[0]),
			&(dataArray->phi[0])
			);
}

void readDataArray(DataArray dataArray, char *fileName, int count) {
	FILE *fp = fopen(fileName, "r");
	if (fp == NULL) {
		eprintf("Error while opening the data array file: %s\n", fileName);
	}
	int i;
	for (i = 0; i < count; i++) {
		fscanf(fp, "%lf %lf %lf %lf %lf %lf",
				&(dataArray[i].x[0]),
				&(dataArray[i].dx[0]),
				&(dataArray[i].y[0]),
				&(dataArray[i].dy[0]),
				&(dataArray[i].delta[0]),
				&(dataArray[i].phi[0])
				);
	}
	fclose(fp);
}

void readDataArrayWithSpin(DataArray dataArray, SpinDataArray spinDataArray, char *fileName, int count) {
	FILE *fp = fopen(fileName, "r");
	if (fp == NULL) {
		eprintf("Error while opening the data array file: %s\n", fileName);
	}
	int i;
	for (i = 0; i < count; i++) {
		fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
				&(dataArray[i].x[0]),
				&(dataArray[i].dx[0]),
				&(dataArray[i].y[0]),
				&(dataArray[i].dy[0]),
				&(dataArray[i].delta[0]),
				&(dataArray[i].phi[0]),
				&(spinDataArray[i].sx[0]),
				&(spinDataArray[i].sy[0]),
				&(spinDataArray[i].sz[0])
				);
	}
	fclose(fp);
}

int main(int argc, char **argv) {
	char fileName[200] = "";
	char inputFileName[200] = "";
	char *outputFileName;
	if (safeMalloc((void**) &outputFileName, 200, sizeof (char))) {
		eprintf("Failed to allocate memory for the output file name");
	}
	outputFileName[0] = '\0';
	int separateFiles = 0, accelerate = 0, calcSpin = 0;
	int argcCounter, particleCount = 1, iterationCount = ITER;
	int xSize, dxSize, ySize, dySize, deltaSize, phiSize;
	int spinSize[3];
	Map x, dx, y, dy, delta, phi;
	Map dev_x, dev_dx, dev_y, dev_dy, dev_delta, dev_phi;
	SpinMap spin;
	SpinMap dev_spin;

	DataArray dataArray;
	DataArray dev_dataArray;
	SpinDataArray spinDataArray;
	SpinDataArray dev_spinDataArray;

	Properties properties;

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
					sscanf(&argv[1][3], "%d", &iterationCount);
					break;

				default:
					eprintf("Wrong Argument: %s\n", argv[1]);
			}
		} else {
			switch (argv[1][1]) {
				case 'f':
					separateFiles = 1;
					break;
				case 's':
					calcSpin = 1;
					break;
				case 'g':
					if (!strstr(&argv[1][2], "pu") || argv[1][4] != '\0') {
						eprintf("Wrong Argument: %s\n", argv[1]);
					} else {
						accelerate = 1;
					}
					break;
				case '-':
					if (!strstr(&argv[1][2], "help") || argv[1][6] != '\0') {
						eprintf("Wrong Argument: %s\n", argv[1]);
					}
				case 'h':
					if (strstr(&argv[1][2], "help") || argv[1][2] == '\0') {
						printf("Calculates a certain amount of steps of a charged particle in a inhomogeneous\nmagnetic field.\n\n");
						printf("<executable> (-h|--help) | <executable> [-m=<mapFileName>] [-c=<coeffFileName>]\n[-o=<outputFileName> [-f]] [-gpu] [-i=<nr>] [-s]\n\n");
						printf("-h, --help\t\t Display help\n");
						printf("-m=<mapFileName>\t Set the map file to be <mapFileName>. If not set, it\n\t\t\t will be asked for in the program itself.\n");
						printf("-c=<coeffFileName>\t Set the coefficients file to be <coeffFileName>. If\n\t\t\t not set, it will be asked for in the program itself.\n\t\t\t Note that the coefficients file supports multiple\n\t\t\t particles, while if the program is run without this\n\t\t\t file, it supports only one particle.\n");
						printf("-o=<outputFileName>\t Set the output file to be <outputFileName>. If not\n\t\t\t set, it will default to stdout\n");
						printf("-f\t\t\t Choose if you want one output file or (if applicable)\n\t\t\t multiple output files. Note that this parameter can\n\t\t\t only be set if an output file is set.\n");
						printf("-gpu\t\t\t Choose if you want to use GPU acceleration. You would\n\t\t\t need a NVIDIA videocard with compute capability 2.0\n\t\t\t or higher (Fermi microarchitecture).\n");
						printf("-i=<nr>\t\t\t Set the number of iterations to <nr>. If not set, it\n\t\t\t will default to 4000.\n");
						printf("-s\t\t\t Choose if you want the spin to be calculated as well.\n\t\t\t At the moment no cpu support\n\n");
						printf("Press Enter to continue...\n");
						getchar();
						exit(EXIT_SUCCESS);
					} else {
						eprintf("Wrong Argument: %s\n", argv[1]);
					}
					break;

				default:
					eprintf("Wrong Argument: %s\n", argv[1]);
			}
		}
		++argv;
		--argcCounter;
	}
	if (separateFiles == 1 && strncmp(outputFileName, "\0", 1) == 0) {
		eprintf("-s shouldn't be used without setting an output file");
	}

	// if not set in argument, ask for file name of the map file
	if (strncmp(fileName, "\0", 1) == 0) {
		iprintf("Filename of the map: ");
		scanf("%s", fileName);
	}

	// open the map file
	iprintf("open file\n");
	FILE *scanFileP = fopen(fileName, "r");
	iprintf("check if file is NULL\n");
	if (scanFileP == NULL) {
		eprintf("Error while opening the map file: %s\n", fileName);
	}
	iprintf("Get map sizes\n");
	char* dummy = (char*) malloc(200 * sizeof (char));

	// skip the particle properties, they aren't needed at this stage
	do {
		dummy = fgets(dummy, 200, scanFileP);
	} while (!strstr(dummy, " A "));

	// scan the maps for their sizes
	iprintf(" x\n");
	scanMap(scanFileP, &xSize);
	iprintf(" dx\n");
	scanMap(scanFileP, &dxSize);
	iprintf(" y\n");
	scanMap(scanFileP, &ySize);
	iprintf(" dy\n");
	scanMap(scanFileP, &dySize);
	iprintf(" delta\n");
	scanMap(scanFileP, &deltaSize);
	iprintf(" phi\n");
	scanMap(scanFileP, &phiSize);
	if (calcSpin) {
		iprintf(" spin\n");
		scanSpinMap(scanFileP, spinSize);
	}
	fclose(scanFileP);
	iprintf("map sizes: %d %d %d %d %d %d\n", xSize, dxSize, ySize, dySize, deltaSize, phiSize);
	if (calcSpin) {
		iprintf("spin map row sizes: %d %d %d\n", spinSize[0], spinSize[1], spinSize[2]);
	}

	// allocate memory for the map
	iprintf("malloc map x\n");
	mallocMap(&x, xSize);
	cudaMallocMap(&dev_x, xSize);
	iprintf("malloc map dx\n");
	mallocMap(&dx, dxSize);
	cudaMallocMap(&dev_dx, dxSize);
	iprintf("malloc map y\n");
	mallocMap(&y, ySize);
	cudaMallocMap(&dev_y, ySize);
	iprintf("malloc map dy\n");
	mallocMap(&dy, dySize);
	cudaMallocMap(&dev_dy, dySize);
	iprintf("malloc map delta\n");
	mallocMap(&delta, deltaSize);
	cudaMallocMap(&dev_delta, deltaSize);
	iprintf("malloc map phi\n");
	mallocMap(&phi, phiSize);
	cudaMallocMap(&dev_phi, phiSize);
	if (calcSpin) {
		iprintf("malloc spin map\n");
		mallocSpinMap(&spin, spinSize[0], spinSize[1], spinSize[2]);
		cudaMallocSpinMap(&dev_spin, spinSize[0], spinSize[1], spinSize[2]);
	}

	// read some variables and the map lines from the map file
	iprintf("open file\n");
	FILE *mapFileP = fopen(fileName, "r");
	iprintf("check if file is NULL\n");
	if (mapFileP == NULL) {
		eprintf("Error while opening the map file: %s\n", fileName);
	}
	iprintf("read properties\n");
	readProperties(mapFileP, &properties);
	iprintf("read map x\n");
	readMap(mapFileP, x, 0);
	iprintf("read map dx\n");
	readMap(mapFileP, dx, 1);
	iprintf("read map y\n");
	readMap(mapFileP, y, 2);
	iprintf("read map dy\n");
	readMap(mapFileP, dy, 3);
	iprintf("read map delta\n");
	readMap(mapFileP, delta, 4);
	iprintf("read map phi\n");
	readMap(mapFileP, phi, 5);
	if (calcSpin) {
		for (int i = 0; i < 4; i++) fgets(dummy, 200, mapFileP);
		free(dummy);
		iprintf("read spin maps");
		readSpinMap(mapFileP, spin->x);
		readSpinMap(mapFileP, spin->y);
		readSpinMap(mapFileP, spin->z);
	}
	fclose(mapFileP);

	// allocate mempry for the data array
	if (strncmp(inputFileName, "\0", 1) == 0) {
		// read the coefficients from user input
		iprintf("malloc data array\n");
		mallocDataArray(&dataArray, iterationCount, particleCount);
		cudaMallocDataArray(&dev_dataArray, iterationCount, particleCount);
		iprintf("read input\n");
		getDataArray(dataArray);
	} else {
		scanDataArray(inputFileName, &particleCount);
		iprintf("malloc data array\n");
		if (calcSpin) {
			mallocSpinDataArray(&spinDataArray, iterationCount, particleCount);
			cudaMallocSpinDataArray(&dev_spinDataArray, iterationCount, particleCount);
		}
		mallocDataArray(&dataArray, iterationCount, particleCount);
		cudaMallocDataArray(&dev_dataArray, iterationCount, particleCount);
		iprintf("read file\n");
		if (calcSpin) {
			readDataArrayWithSpin(dataArray, spinDataArray, inputFileName, particleCount);
		} else {
			readDataArray(dataArray, inputFileName, particleCount);
		}
		iprintf("Particle count: %d\n", particleCount);
	}

	// calculate the coefficients for 4000 iterations
	if (!accelerate) {
		//cpu
		if (calcSpin) {
			spinKernel(dataArray, spinDataArray, x, dx, y, dy, delta, phi, spin, particleCount, iterationCount);
		} else {
			kernel(dataArray, x, dx, y, dy, delta, phi, particleCount, iterationCount);
		}
	} else {
		//gpu
		cudaEventRecord(start);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, 0);
		int blocks = MIN(1048576, particleCount);
		iprintf("Device name: %s\n", deviceProperties.name);
		iprintf("Used blocks: %d\n", blocks);
		int dimBlocks = sqrt((float) blocks);
		int dimThreads = blocks / dimBlocks + (blocks % dimBlocks > 0);
		cudaMemcpyMap(dev_x, x, cudaMemcpyHostToDevice);
		cudaMemcpyMap(dev_dx, dx, cudaMemcpyHostToDevice);
		cudaMemcpyMap(dev_y, y, cudaMemcpyHostToDevice);
		cudaMemcpyMap(dev_dy, dy, cudaMemcpyHostToDevice);
		cudaMemcpyMap(dev_delta, delta, cudaMemcpyHostToDevice);
		cudaMemcpyMap(dev_phi, phi, cudaMemcpyHostToDevice);
		cudaMemcpyFirstDataArray(dev_dataArray, dataArray, particleCount);
		if (calcSpin) {
			//cudaMemcpySpinMap(dev_spin, spin, cudaMemcpyHostToDevice);
			cudaMemcpyFirstSpinDataArray(dev_spinDataArray, spinDataArray, particleCount);
		}

		cudaEventRecord(stopH2D);

		if (calcSpin) {
			//cudaSpinKernel<<<dimBlocks, dimThreads>>>(dev_dataArray, dev_spinDataArray, dev_x, dev_dx, dev_y, dev_dy, dev_delta, dev_phi, dev_spin, particleCount, iterationCount);
		} else {
			cudaKernel << <dimBlocks, dimThreads>>>(dev_dataArray, dev_x, dev_dx, dev_y, dev_dy, dev_delta, dev_phi, particleCount, iterationCount);
		}


		cudaEventRecord(stopK);
	}
	for (int n = 0; n < particleCount; n++) {
		// if acceleration is on, copy input from device to host
		if (accelerate) {
			cudaMemcpyDataArray(&(dataArray[n]), &(dev_dataArray[n]), cudaMemcpyDeviceToHost);
			if (calcSpin) {
				cudaMemcpySpinDataArray(&(spinDataArray[n]), &(dev_spinDataArray[n]), cudaMemcpyDeviceToHost);
			}
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
				eprintf("Error while opening the output file: %s\n", fullOutputFileName);
			}
		} else {
			outputFile = stdout;
		}
		if (calcSpin) {
			for (int i = 0; i < iterationCount; i++) {
				fprintf(outputFile, "%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n", dataArray[n].x[i], dataArray[n].dx[i], dataArray[n].y[i], dataArray[n].dy[i], dataArray[n].delta[i], dataArray[n].phi[i], spinDataArray[n].sx[i], spinDataArray[n].sy[i], spinDataArray[n].sz[i]);
			}
		} else {
			for (int i = 0; i < iterationCount; i++) {
				fprintf(outputFile, "%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n", dataArray[n].x[i], dataArray[n].dx[i], dataArray[n].y[i], dataArray[n].dy[i], dataArray[n].delta[i], dataArray[n].phi[i]);
			}
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


	iprintf("Elapsed time:\n\n");
	iprintf(" H2D:    %f\n", h2dTime);
	iprintf(" Kernel: %f\n", kTime);
	iprintf(" D2H:    %f\n", d2hTime);

	// clean up the heap and tell that the computation is finished
	iprintf("free map x\n");
	freeMap(&x);
	cudaFreeMap(&dev_x);
	iprintf("free map dx\n");
	freeMap(&dx);
	cudaFreeMap(&dev_dx);
	iprintf("free map y\n");
	freeMap(&y);
	cudaFreeMap(&dev_y);
	iprintf("free map dy\n");
	freeMap(&dy);
	cudaFreeMap(&dev_dy);
	iprintf("free map delta\n");
	freeMap(&delta);
	cudaFreeMap(&dev_delta);
	iprintf("free map phi\n");
	freeMap(&phi);
	cudaFreeMap(&dev_phi);
	if (calcSpin) {
		iprintf("free spin map\n");
		freeSpinMap(&spin);
	}

	iprintf("free data array\n");
	freeDataArray(&dataArray, particleCount);
	cudaFreeDataArray(&dev_dataArray, particleCount);
	if (calcSpin) {
		iprintf("free spin data array\n");
		freeSpinDataArray(&spinDataArray, particleCount);
	}

	if (safeFree((void**) &outputFileName)) {
		wprintf("failed to free the variable containing the output file name");
	}
	iprintf("Output is created. Press Enter to continue...\n");
	getchar();
	if (strncmp(inputFileName, "\0", 1) == 0) {
		getchar();
	}
	return 0;
}
