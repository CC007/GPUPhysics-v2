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
CND_CONF=CUDA-SDK
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/cc/main.o


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
LDLIBSOPTIONS=-L../../NVIDIA_GPU_Computing_SDK/C/lib -L../../NVIDIA_GPU_Computing_SDK/shared/lib -lcudart -lcutil_x86_64

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gpuphysics-v2 ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/src/cc/main.o: src/cc/main.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/cc
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I../../NVIDIA_GPU_Computing_SDK/shared/inc -I../../NVIDIA_GPU_Computing_SDK/C/common/inc/ -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/src/cc/main.o src/cc/main.cpp

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
