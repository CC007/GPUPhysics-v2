/* 
 * File:   extendedio.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on Oktober 4, 2015, 3:23 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "../../include/safemem.h"

void eprintf(const char *format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, " ERROR : ");
	vfprintf(stderr, format, args);
	va_end(args);
	exit(EXIT_FAILURE);
}

void wprintf(const char *format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "WARNING: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

void iprintf(const char *format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "  INFO : ");
	vfprintf(stderr, format, args);
	va_end(args);
}

void printBinaryDouble(FILE *stream, double d) {
	char *dChars;
	if (safeMalloc((void**) &dChars, 1, sizeof (d) )) {
		eprintf("Cannot allocate memory for printing doubles as binary");
	}
	memcpy(dChars, &d, sizeof (d));
	for(int i=0;i<sizeof(d);i++){
		putc(dChars[i], stream);
	}
}
