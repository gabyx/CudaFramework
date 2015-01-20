#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


#Detect Build Type if Debug appears in CMAKE_BINARY_DIR path
STRING(REGEX MATCH ".*(Debug|debug).*" DEBUG_MATCH ${CMAKE_BINARY_DIR} )
if(DEBUG_MATCH)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "The build type for makefile based generators")
    SET(PROJECTNAMEPREFIX Debug)
ELSE()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "The build type for makefile based generators")
    SET(PROJECTNAMEPREFIX Release)
ENDIF()


set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")
message(STATUS "Module path is now set to: " ${CMAKE_MODULE_PATH} )


#The name of the project
set(PROJECTNAME "CudaFramework")
if(${CMAKE_GENERATOR} MATCHES "Unix Makefiles")
        set(PROJECTNAME  ${PROJECTNAME}${PROJECTNAMEPREFIX})
endif()
PROJECT(${PROJECTNAME})

SET(CudaFramework_BINARY_DIR ${PROJECT_BINARY_DIR})
SET(CudaFramework_ROOT_DIR   ${PROJECT_SOURCE_DIR})


#Add some multithreaded build support =====================================================================================================
MARK_AS_ADVANCED(MULTITHREADED_BUILD)
SET(MULTITHREADED_BUILD ON CACHE BOOL "Parallel build with as many threads as possible!")
if(MULTITHREADED_BUILD)
	if(${CMAKE_GENERATOR} MATCHES "Unix Makefiles")
            file(COPY ${CudaFramework_ROOT_DIR}/cmake/parallelmake.sh DESTINATION ${PROJECT_BINARY_DIR}
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
                NO_SOURCE_PERMISSIONS
            )
            SET(CMAKE_MAKE_PROGRAM "${PROJECT_BINARY_DIR}/parallelmake.sh")
            MESSAGE(STATUS "Set make program to ${PROJECT_BINARY_DIR}/parallelmake.sh")
    elseif(MSVC)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "/MP")
      MESSAGE(STATUS "Added parallel build arguments to CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    endif()
endif()
# ========================================================================================================================================

MESSAGE(STATUS "Compiler ID is: " ${CMAKE_CXX_COMPILER_ID})
IF(NOT CudaFramework_SET_COMPILER_FLAGS_INTERNAL)
	IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")

		SET(CudaFramework_SET_COMPILER_FLAGS_INTERNAL TRUE CACHE INTERNAL "x" FORCE)
		MESSAGE(STATUS "Setting Values for GNU")
		SET(CMAKE_C_FLAGS "-fmessage-length=0" CACHE STRING "Flags for C Compiler" FORCE)
		SET(CMAKE_CXX_FLAGS "-std=c++11 -Wno-enum-compare" CACHE STRING "Flags for CXX Compiler" FORCE)
		SET(CMAKE_CXX_FLAGS_DEBUG          "-g" CACHE STRING "Flags for CXX Compiler for debug builds" FORCE)
    
	ELSEIF ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" )

		MESSAGE(STATUS "Setting Values for Clang")
		SET(CudaFramework_SET_COMPILER_FLAGS_INTERNAL TRUE CACHE INTERNAL "x" FORCE)
		SET(CMAKE_C_FLAGS                "-Wall" CACHE STRING "Flags for C Compiler" FORCE)
		SET(CMAKE_C_FLAGS_DEBUG          "-g" CACHE STRING "Flags for C Compiler for debug builds" FORCE)
		SET(CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Flags for C Compiler for release minsize builds" FORCE)
		SET(CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG" CACHE STRING "Flags for C Compiler for release builds" FORCE)
		SET(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "Flags for C Compiler for release builds with debug info" FORCE)

        #SET (CMAKE_CXX_FLAGS             "-std=c++11 -ferror-limit=50 -Weverything -Wno-extra-semi -Wno-documentation -pedantic -Wno-conversion -Wno-comment -Wno-unused-parameter -Wno-deprecated-register -Wno-float-equal -Wno-switch -fdiagnostics-show-template-tree -Wno-c++98 -Wno-c++98-compat-pedantic -Wno-deprecated" CACHE STRING "" FORCE)
		SET(CMAKE_CXX_FLAGS                "-std=c++11 -ferror-limit=50 -w" CACHE STRING "Flags for CXX Compiler" FORCE)
		SET(CMAKE_CXX_FLAGS_DEBUG          "-g " CACHE STRING "Flags for CXX Compiler for debug builds" FORCE)
		SET(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Flags for CXX Compiler for release minsize builds" FORCE)
		SET(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG" CACHE STRING "Flags for CXX Compiler for release builds" FORCE)
		SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "Flags for CXX Compiler for release builds with debug info" FORCE)

        SET(CMAKE_AR      "/usr/bin/local/llvm-ar" CACHE STRING "archiver" FORCE )
        SET(CMAKE_LINKER  "/usr/bin/local/llvm-ld" CACHE STRING "linker tool" FORCE )
        SET(CMAKE_NM      "/usr/bin/local/llvm-nm" CACHE STRING "nm tool" FORCE )
        SET(CMAKE_OBJDUMP "/usr/bin/local/llvm-objdump" CACHE STRING "objdump tool" FORCE )
        SET(CMAKE_RANLIB  "/usr/bin/local/llvm-ranlib" CACHE STRING "ranlib tool" FORCE )

	ENDIF()
    MESSAGE(WARNING "CudaFramework: Memory sanitizer -fsanitize=address and memory leak detection-fsanitize=leak disabled in Debug because of Bug with CUDA")

ENDIF()


# Searching CUDA and Flags ======================================================
FIND_PACKAGE(CUDA REQUIRED)

IF(NOT CudaFramework_SET_COMPILER_CUDA_FLAGS_INTERNAL)
    SET(CudaFramework_SET_COMPILER_CUDA_FLAGS_INTERNAL TRUE CACHE INTERNAL "x" FORCE)
    SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "Verbose build" FORCE)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF CACHE BOOL "Propagate C/CXX Flags and friends to the host compiler in NVCC via -Xompile" FORCE)

    # Set the architecture for the grafics card!
    SET(CUDA_NVCC_FLAGS
    "-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\""
    "-gencode=arch=compute_30,code=\\\"sm_30,compute_30\\\""
    "-gencode=arch=compute_35,code=\\\"sm_35,compute_35\\\""
    "--ptxas-options=-v"
    CACHE STRING "Semicolon delimited several arguments for the NVCC compiler" FORCE)

    SET(CUDA_NVCC_FLAGS_DEBUG
    "-g"
    "-G"
    CACHE STRING "Semicolon delimited several arguments for the NVCC compiler" FORCE)

    SET(CUDA_NVCC_FLAGS_RELEASE "-O2"
    CACHE STRING "Semicolon delimited several arguments for the NVCC compiler" FORCE)
ENDIF()
#===================

#Searching Eigen ==========================
FIND_PACKAGE(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
#==========================================

##Searching Boost ===========================================================================================================
## Is not needed, only the format library is needed (headers only)
#set(Boost_ADDITIONAL_VERSIONS "1.46" "1.46.0" "1.46.1" "1.45" "1.45.0" "1.44" "1.44.0" "1.42" "1.42.0" "1.41.0" "1.41" "1.40.0" "1.40" "1.39.0" "1.39" "1.38.0" "1.38" "1.37.0" "1.37" )
##set(BOOST_COMPONENTS system)
#find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
#if (NOT Boost_FOUND)
	## Try again with the other type of libs
	#if(Boost_USE_STATIC_LIBS)
		#set(Boost_USE_STATIC_LIBS)
	#else()
		#set(Boost_USE_STATIC_LIBS ON)
	#endif()
	#find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
#endif()
## ==================================================================================================================

# BLAS VARIANT ====================================================================================================
SET(CudaFramework_USE_BLAS_VARIANT "NONE" CACHE STRING "Blas Variant either 'INTEL', 'GOTO','OPENBLAS', 'NONE' ")
MARK_AS_ADVANCED(CudaFramework_USE_BLAS_VARIANT)
STRING(COMPARE EQUAL ${CudaFramework_USE_BLAS_VARIANT} "GOTO" USE_GOTO_BLAS_FLAG)
STRING(COMPARE EQUAL ${CudaFramework_USE_BLAS_VARIANT} "OPENBLAS" USE_OPEN_BLAS_FLAG)
STRING(COMPARE EQUAL ${CudaFramework_USE_BLAS_VARIANT} "INTEL" USE_INTEL_BLAS_FLAG)

if(USE_INTEL_BLAS_FLAG)
    MESSAGE(FATAL_ERROR "Intel Blas Configuration not yet scripted in cmake")
endif(USE_INTEL_BLAS_FLAG)

if(USE_GOTO_BLAS_FLAG)
    MESSAGE(FATAL_ERROR "Goto Blas Configuration not yet scripted in cmake")
endif(USE_GOTO_BLAS_FLAG)


if(USE_OPEN_BLAS_FLAG)
	FIND_PACKAGE(OpenBlas)
	if(OPENBLAS_FOUND)
		SET(BLAS_INCLUDE_DIR ${OPENBLAS_INCLUDE_DIR} CACHE STRING "Blas include directory")
		SET(BLAS_LIBS  ${OPENBLAS_LIBRARY} CACHE STRING "Blas library")
        SET(BLAS_CXX_FLAGS "" CACHE STRING "Blas CXX compiler options")
	endif()
endif()
# ==================================================================================================================


MARK_AS_ADVANCED( CMAKE_DEBUG_POSTFIX )
SET(CMAKE_DEBUG_POSTFIX "-dbg" CACHE STRING "Debug postfix for library/executable")

# Common Source directory
set(COMMON_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
message(STATUS "CommonSource Dir is set to: ${COMMON_SOURCE_DIR} ")

# WRITE CONFIGURATION FILE
INCLUDE(${CudaFramework_ROOT_DIR}/cmake/WriteConfigFile.cmake)
SET(CudaFramework_CONFIG_FILE ${CudaFramework_BINARY_DIR}/include/CudaFramework/General/ConfigureFile.hpp) 
CudaFramework_WRITE_CONFIG_FILE( ${CudaFramework_CONFIG_FILE} ${CudaFramework_ROOT_DIR})
#=========================

# Set common includes for all projects which need them
INCLUDE(IncludeModulesCudaFramework)

INCLUDE_GENERAL_CUDA(GeneralCommonFiles_SRC GeneralCommonFiles_INC GeneralCommonFiles_INC_DIRS ${COMMON_SOURCE_DIR})

INCLUDE_GAUSS_SEIDEL_CUDA(GaussSeidelGPU_SRC GaussSeidelGPU_INC GaussSeidelGPU_INC_DIRS ${COMMON_SOURCE_DIR})
INCLUDE_PROX_CUDA(ProxGPU_SRC ProxGPU_INC ProxGPU_INC_DIRS ${COMMON_SOURCE_DIR})
INCLUDE_VECTOR_ADD_CUDA(VectorAddGPU_SRC VectorAddGPU_INC VectorAddGPU_INC_DIRS ${COMMON_SOURCE_DIR})
INCLUDE_MATRIX_MULT_CUDA(MatrixMultGPU_SRC MatrixMultGPU_INC MatrixMultGPU_INC_DIRS ${COMMON_SOURCE_DIR})
INCLUDE_MATRIX_VECTOR_MULT_CUDA(MatrixVectorMultGPU_SRC MatrixVectorMultGPU_INC MatrixVectorMultGPU_INC_DIRS ${COMMON_SOURCE_DIR})
INCLUDE_TESTS_CUDA(TestsGPU_SRC TestsGPU_INC TestsGPU_INC_DIRS ${COMMON_SOURCE_DIR})


set(COMMON_FILES_INC
	${GeneralCommonFiles_INC}
)
set(COMMON_FILES_SRC
	${GeneralCommonFiles_SRC}
)

set(COMMON_INCLUDE_DIR
	${GeneralCommonFiles_INC_DIRS}
    ${PROJECT_BINARY_DIR}/include
    
    ${GaussSeidelGPU_INC_DIRS}
    ${ProxGPU_INC_DIRS}
    ${VectorAddGPU_INC_DIRS}
    ${MatrixMultGPU_INC_DIRS}
    ${MatrixVectorMultGPU_INC_DIRS}
    ${TestsGPU_INC_DIRS}
)

# =============================================================================================================

add_subdirectory(projects/TestCudaModern)
add_subdirectory(projects/TestIntrusivePtr)
add_subdirectory(projects/TestTemplateKernel)
add_subdirectory(projects/TestMatrix)
#add_subdirectory(TestMatrixSDK)
add_subdirectory(projects/TestGaussSeidel)
add_subdirectory(projects/PerformanceMatrix)
add_subdirectory(projects/PerformanceVector)
add_subdirectory(projects/PerformanceProx)

