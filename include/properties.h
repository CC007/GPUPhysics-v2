/* 
 * File:   properties.h
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on April 27, 2015, 7:56 PM
 */

#ifndef PROPERTIES_H
#define	PROPERTIES_H

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

#endif	/* PROPERTIES_H */

