#The name of the project
PROJECT(TestCudaCmake)

message(STATUS "==============================Project: ${PROJECT_NAME} ==================================")
include(PrintListMacro)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#include directories
set(INCLUDE_DIRS 
${PROJECT_SOURCE_DIR}/src
${COMMON_INCLUDE_DIR}
)
include_directories(${INCLUDE_DIRS})
PRINTLIST("Include Directories are:" "${INCLUDE_DIRS}")

#link directories
set(LINK_DIRS
)
link_directories(${LINK_DIRS})
PRINTLIST("Link directories are:" "${LINK_DIRS}")

#The rule to build the executable program
CUDA_ADD_EXECUTABLE(CmakeCudaTest src/test.cu)
target_link_libraries(CmakeCudaTest)

message(STATUS "=========================================================================")
