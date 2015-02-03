#The name of the project
PROJECT(JORProxVel)

message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#include directories
set(INCLUDE_DIRS
       # ${PROJECT_SOURCE_DIR}/inc

        ${ReductionKernel_INCLUDE_DIR}/HostUtilities/
        ${ReductionKernel_INCLUDE_DIR}/DeviceUtilities/
        ${ReductionKernel_INCLUDE_DIR}/DeviceFunctions/
        ${ReductionKernel_INCLUDE_DIR}/KernelsWrappers/

        ${BodyInitKernel_INCLUDE_DIR}
        ${ContactInitKernel_INCLUDE_DIR}
        ${ConvergenceCheckKernel_INCLUDE_DIR}
        ${ReductionKernel_INCLUDE_DIR}
        ${JORProxVelKernel_INCLUDE_DIR}
        ${ContactIterationKernel_INCLUDE_DIR}
        ${JORProxVelCommon_INCLUDE_DIRS}


        ${COMMON_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
)

include_directories(${INCLUDE_DIRS})
PRINTLIST("Include Directories are:" "${INCLUDE_DIRS}")


#src directories
set(SOURCE_FILES
      ${COMMON_FILES_SRC}
      ${PROJECT_SOURCE_DIR}/src/main.cpp

      ${JORProxVelKernel_SRC}
      ${BodyInitKernel_SRC}
      ${ContactInitKernel_SRC}
      ${ConvergenceCheckKernel_SRC}
      ${ReductionKernel_SRC}

      ${ContactIterationKernel_SRC}


)

#src directories
set(INCLUDE_FILES
      ${COMMON_FILES_INC}
      
      ${JORProxVelKernel_INC}
      ${BodyInitKernel_INC}
      ${ContactInitKernel_INC}
      ${ConvergenceCheckKernel_INC}
      ${ReductionKernel_INC}
      ${ContactIterationKernel_INC}
      ${JORProxVelCommon_INC}


)

PRINTLIST("Source Files:" "${SOURCE_FILES}")


#link directories
set(LINK_DIRS ${Boost_LIBRARY_DIRS})
link_directories(${LINK_DIRS})
PRINTLIST("Link directories are:" "${LINK_DIRS}")

set(LINKED_LIBS  ${Boost_LIBRARIES})
PRINTLIST("Linked libraries are:" "${LINKED_LIBS}")

#Adding src directory to the project
#The rule to build the executable program
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} )
target_link_libraries(${PROJECT_NAME}  ${LINKED_LIBS})

message(STATUS "=========================================================================")
