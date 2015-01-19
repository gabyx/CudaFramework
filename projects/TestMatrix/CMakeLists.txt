
#The name of the project
PROJECT(TestMatrix)
message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


#Append includes
string(REPLACE "/C" "/" NVIDIA_SDK_DIR "${CUDA_SDK_ROOT_DIR}")
message(STATUS "NVIDIA Sample ROOT Dir is: ${NVIDIA_SDK_DIR}")

#include directories
set(INCLUDE_DIRS 
    ${Boost_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/inc
    ${COMMON_INCLUDE_DIR}
    ${STLSOFT_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${BLAS_INCLUDE_DIR}
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
set(LINKED_LIBS  ${CUDA_cublas_LIBRARY} )
PRINTLIST("Linked libraries are:" "${LINKED_LIBS}")



set(SOURCE_FILES 
	"src/main.cpp" 
	${COMMON_FILES_SRC}
	${MatrixMultGPU_SRC}
    ${MatrixVectorMultGPU_SRC}
	${VectorAddGPU_SRC}
)

set(INCLUDE_FILES
	${COMMON_FILES_INC}
	${MatrixMultGPU_INC}
    ${MatrixVectorMultGPU_INC}
	${VectorAddGPU_INC}
	${ProxGPU_INC}
) 

SET(CMAKE_DEBUG_POSTFIX "D")

set_source_files_properties( INCLUDE_FILES PROPERTIES HEADER_FILE_ONLY TRUE)


#The rule to build the executable program
IF(MSVC)
make_groups_common_files()
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} )
ELSE(MSVC)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}  ${SOURCE_FILES} ${INCLUDE_FILES} )
ENDIF(MSVC)
target_link_libraries(${PROJECT_NAME}  ${LINKED_LIBS})

message(STATUS "=========================================================================")
