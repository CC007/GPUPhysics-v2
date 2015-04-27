/* 
 * File:   params.h
 * Author: rik
 *
 * Created on April 27, 2015, 7:56 PM
 */

#ifndef PARAMS_H
#define	PARAMS_H

#ifdef	__cplusplus
extern "C" {
#endif

    typedef struct Properties {
        double mass;
        double momentum;
        double kinEn;
        double gamma;
        double beta;
        double mAnomalyG;
        double spinTuneGgamma;
        double lRefOrbit;
    } Properties;

#ifdef	__cplusplus
}
#endif

#endif	/* PARAMS_H */

