#The name of the project
PROJECT(TestMatrixSDK)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Append includes
string(REPLACE "/C" "/" NVIDIA_SDK_DIR "${CUDA_SDK_ROOT_DIR}")
message("NVIDIA Sample ROOT Dir is")
message("${NVIDIA_SDK_DIR}")

#include directories
include_directories(
"${NVIDIA_SDK_DIR}/C/common/inc"
"${NVIDIA_SDK_DIR}/shared/inc"
"${PROJECT_SOURCE_DIR}/src"
)

#link directories
link_directories(
"${NVIDIA_SDK_DIR}/C/common/lib"
"${NVIDIA_SDK_DIR}/shared/lib"
)

#set_property(CACHE CUDA_NVCC_FLAGS PROPERTY VALUE "-arch=sm_20"; -Xcompiler /MDd)
set_property(CACHE CUDA_VERBOSE_BUILD PROPERTY VALUE ON)
message("${CUDA_NVCC_FLAGS}")

#Adding src directory to the project
ADD_SUBDIRECTORY(src)


