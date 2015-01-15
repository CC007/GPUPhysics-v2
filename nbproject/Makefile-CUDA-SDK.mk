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
FC=
AS=

# Macros
CND_PLATFORM=GNU+CUDA-Linux-x86
CND_CONF=CUDA-SDK
CND_DISTDIR=dist

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/main.o


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
	"${MAKE}"  -f nbproject/Makefile-CUDA-SDK.mk dist/CUDA-SDK/GNU+CUDA-Linux-x86/cuda-project

dist/CUDA-SDK/GNU+CUDA-Linux-x86/cuda-project: ${OBJECTFILES}
	${MKDIR} -p dist/CUDA-SDK/GNU+CUDA-Linux-x86
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cuda-project ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I../../NVIDIA_GPU_Computing_SDK/shared/inc -I../../NVIDIA_GPU_Computing_SDK/C/common/inc/ -Xcompiler "-MMD -MP -MF $@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/CUDA-SDK
	${RM} dist/CUDA-SDK/GNU+CUDA-Linux-x86/cuda-project

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
