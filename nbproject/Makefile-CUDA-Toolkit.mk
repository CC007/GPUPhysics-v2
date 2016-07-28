#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=
AS=

# Macros
CND_PLATFORM=mpiccc-Windows
CND_DLIB_EXT=dll
CND_CONF=CUDA-Toolkit
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/c/data.o \
	${OBJECTDIR}/src/c/extendedio.o \
	${OBJECTDIR}/src/c/kernel.o \
	${OBJECTDIR}/src/c/map.o \
	${OBJECTDIR}/src/c/properties.o \
	${OBJECTDIR}/src/c/safemem.o \
	${OBJECTDIR}/src/c/spinmap.o \
	${OBJECTDIR}/src/cu/data.o \
	${OBJECTDIR}/src/cu/kernel.o \
	${OBJECTDIR}/src/cu/map.o \
	${OBJECTDIR}/src/cu/physicsSimulation.o \
	${OBJECTDIR}/src/cu/safemem.o \
	${OBJECTDIR}/src/cu/spinmap.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=../../NVIDIA_GPU_Computing_SDK/C/lib ../../NVIDIA_GPU_Computing_SDK/shared/lib cudart

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2.exe

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2.exe: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2 ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/src/c/data.o: src/c/data.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/data.o src/c/data.c

${OBJECTDIR}/src/c/extendedio.o: src/c/extendedio.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/extendedio.o src/c/extendedio.c

${OBJECTDIR}/src/c/kernel.o: src/c/kernel.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/kernel.o src/c/kernel.c

${OBJECTDIR}/src/c/map.o: src/c/map.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/map.o src/c/map.c

${OBJECTDIR}/src/c/properties.o: src/c/properties.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/properties.o src/c/properties.c

${OBJECTDIR}/src/c/safemem.o: src/c/safemem.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/safemem.o src/c/safemem.c

${OBJECTDIR}/src/c/spinmap.o: src/c/spinmap.c 
	${MKDIR} -p ${OBJECTDIR}/src/c
	$(COMPILE.c) ${OBJECTDIR}/src/c/spinmap.o src/c/spinmap.c

${OBJECTDIR}/src/cu/data.o: src/cu/data.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.cc) ${OBJECTDIR}/src/cu/data.o src/cu/data.cu

${OBJECTDIR}/src/cu/kernel.o: src/cu/kernel.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.c) ${OBJECTDIR}/src/cu/kernel.o src/cu/kernel.cu

${OBJECTDIR}/src/cu/map.o: src/cu/map.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.cc) ${OBJECTDIR}/src/cu/map.o src/cu/map.cu

${OBJECTDIR}/src/cu/physicsSimulation.o: src/cu/physicsSimulation.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.c) ${OBJECTDIR}/src/cu/physicsSimulation.o src/cu/physicsSimulation.cu

${OBJECTDIR}/src/cu/safemem.o: src/cu/safemem.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.cc) ${OBJECTDIR}/src/cu/safemem.o src/cu/safemem.cu

${OBJECTDIR}/src/cu/spinmap.o: src/cu/spinmap.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	$(COMPILE.c) ${OBJECTDIR}/src/cu/spinmap.o src/cu/spinmap.cu

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2.exe

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
