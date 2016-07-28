/* 
 * File:   extendedio.h
 * Author: Rik Schaaf aka CC007 (http://coolcat007.nl/)
 *
 * Created on October 4, 2015, 3:23 PM
 */


#ifndef EXTENDEDIO_H
#define	EXTENDEDIO_H

#ifdef	__cplusplus
extern "C" {
#endif
#include <stdio.h>

	void eprintf(const char *format, ... );
	void wprintf(const char *format, ... );
	void iprintf(const char *format, ... );
	void printBinaryDouble(FILE *stream, double d);

#ifdef	__cplusplus
}
#endif

#endif	/* EXTENDEDIO_H */

