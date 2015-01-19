
#The name of the project
PROJECT(TestTemplateKernel)
message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


#include directories
set(INCLUDE_DIRS 
${Boost_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}/inc
${PROJECT_SOURCE_DIR}/src
${COMMON_INCLUDE_DIR}
${EIGEN_INCLUDE_DIR}
)
include_directories(${INCLUDE_DIRS})
PRINTLIST("Include Directories are:" "${INCLUDE_DIRS}")

#link directories
set(LINK_DIRS
${BLAS_LIB_DIR}
${Boost_LIBRARY_DIRS}
)
link_directories(${LINK_DIRS})
PRINTLIST("Link directories are:" "${LINK_DIRS}")
set(LINKED_LIBS  ${CUDA_cublas_LIBRARY})
PRINTLIST("Linked libraries are:" "${LINKED_LIBS}")



set(SOURCE_FILES 
	"src/main.cpp"
	"src/call_kernel.cu"
)

set(INCLUDE_FILES
	"inc/main.hpp"
	"inc/kernel.cuh"
) 

set_source_files_properties( INCLUDE_FILES PROPERTIES HEADER_FILE_ONLY TRUE)

#The rule to build the executable program
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}  ${SOURCE_FILES} ${INCLUDE_FILES} )
target_link_libraries(${PROJECT_NAME}  ${LINKED_LIBS})

message(STATUS "=========================================================================")
