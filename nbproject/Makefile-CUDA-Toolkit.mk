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
CC=nvcc
CCC=nvcc
CXX=nvcc
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU+CUDA-Linux-x86
CND_DLIB_EXT=so
CND_CONF=CUDA-Toolkit
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/cc/data.o \
	${OBJECTDIR}/src/cc/map.o \
	${OBJECTDIR}/src/cc/properties.o \
	${OBJECTDIR}/src/cu/oldkernel.o \
	${OBJECTDIR}/src/cu/physicsSimulation.o


# C Compiler Flags
CFLAGS=-m64

# CC Compiler Flags
CCFLAGS=-m64
CXXFLAGS=-m64

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L../../NVIDIA_GPU_Computing_SDK/C/lib -L../../NVIDIA_GPU_Computing_SDK/shared/lib -lcudart

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.c} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2 ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/src/cc/data.o: src/cc/data.c 
	${MKDIR} -p ${OBJECTDIR}/src/cc
	${RM} "$@.d"
	$(COMPILE.c) -O2 -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cc/data.o src/cc/data.c

${OBJECTDIR}/src/cc/map.o: src/cc/map.c 
	${MKDIR} -p ${OBJECTDIR}/src/cc
	${RM} "$@.d"
	$(COMPILE.c) -O2 -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cc/map.o src/cc/map.c

${OBJECTDIR}/src/cc/properties.o: src/cc/properties.c 
	${MKDIR} -p ${OBJECTDIR}/src/cc
	${RM} "$@.d"
	$(COMPILE.c) -O2 -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cc/properties.o src/cc/properties.c

${OBJECTDIR}/src/cu/oldkernel.o: src/cu/oldkernel.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	${RM} "$@.d"
	$(COMPILE.c) -O2 -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cu/oldkernel.o src/cu/oldkernel.cu

${OBJECTDIR}/src/cu/physicsSimulation.o: src/cu/physicsSimulation.cu 
	${MKDIR} -p ${OBJECTDIR}/src/cu
	${RM} "$@.d"
	$(COMPILE.c) -O2 -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cu/physicsSimulation.o src/cu/physicsSimulation.cu

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
