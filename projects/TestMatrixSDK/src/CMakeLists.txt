#Linking with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil")

#Include the FindCUDA script
INCLUDE(FindCUDA)

#The rule to build the executable program matrixMul from matrixmul.cu
#and matrixMul_gold.cpp
CUDA_ADD_EXECUTABLE(matrixMul matrixMul.cu matrixMul_gold.cpp matrixMul.h)

target_link_libraries(matrixMul cutil64D shrUtils64D)