
#The name of the project
PROJECT(PerformanceVector)
message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#include directories
set(INCLUDE_DIRS 
${Boost_INCLUDE_DIR}
${PROJECT_SOURCE_DIR}/inc
${COMMON_INCLUDE_DIR}
${STLSOFT_INCLUDE_DIR}
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
	${GeneralCommonFiles_SRC}
	${MatrixMultGPU_SRC}
	${VectorAddGPU_SRC}
)

set(INCLUDE_FILES
	${GeneralCommonFiles_INC}
	${MatrixMultGPU_INC}
	${VectorAddGPU_INC}
) 

set_source_files_properties( INCLUDE_FILES PROPERTIES HEADER_FILE_ONLY TRUE)


#The rule to build the executable program
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}  ${SOURCE_FILES} ${INCLUDE_FILES} )
target_link_libraries(${PROJECT_NAME}  ${LINKED_LIBS})

message(STATUS "=========================================================================")
