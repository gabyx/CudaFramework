
MACRO(INCLUDE_GAUSS_SEIDEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelTestVariant.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/KernelsGaussSeidel.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_GAUSS_SEIDEL_CUDA)


MACRO(INCLUDE_PROX_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxTestVariant.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxGPU.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxSettings.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/KernelsProx.cuh
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/SorProxGPUVariant.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/JorProxGPUVariant.hpp
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/ProxGPU/ProxGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/ProxGPU/ProxGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_PROX_CUDA)

MACRO(INCLUDE_VECTOR_ADD_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/VectorAddGPU/KernelsVectorAdd.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_VECTOR_ADD_CUDA)

MACRO(INCLUDE_MATRIX_MULT_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixMultGPU/KernelsMatrixMult.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_MATRIX_MULT_CUDA)

MACRO(INCLUDE_MATRIX_VECTOR_MULT_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.hpp
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultTestVariant.hpp
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/KernelsMatrixVectorMult.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/)
endmacro(INCLUDE_MATRIX_VECTOR_MULT_CUDA)

MACRO(INCLUDE_TESTS_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
   	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/TestsGPU/TestsGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/TestsGPU/KernelsTests.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/TestsGPU/TestsGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/TestsGPU/TestsGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_TESTS_CUDA)

MACRO(INCLUDE_GENERAL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )
set(${INC}
	${PROJECT_BINARY_DIR}/include/CudaFramework/General/ConfigureFile.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/Utilities.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FloatingPointType.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/GPUMutex.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/TypeTraitsHelper.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/StaticAssert.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FlopsCounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/AssertionDebug.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/AssertionDebugC.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaEvent.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTimer.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaAlloc.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaRefcounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMemory.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTypeDefs.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContextGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContext.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDevice.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaException.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMemSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrixSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaUtilities.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/Exception.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaError.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrix.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaPrint.hpp
    
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/PerformanceTest.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/KernelTestMethod.hpp
    
    ${COMMON_SOURCE_DIR}/external/pugixml/src/pugixml.hpp
)
set(${SRC}
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaCompilerVersion.cu
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaMemSupport.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDevice.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDeviceGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContextGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContext.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaAlloc.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaUtilities.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaEvent.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaTimer.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaPrint.cpp
	${COMMON_SOURCE_DIR}/src/CudaFramework/General/Utilities.cpp
    
    ${COMMON_SOURCE_DIR}/external/pugixml/src/pugixml.cpp
)
set(${INCLUDE_DIRS} 
    ${COMMON_SOURCE_DIR}/include/
    ${COMMON_SOURCE_DIR}/external/pugixml/src/
)
endmacro(INCLUDE_GENERAL_CUDA)

MACRO(INCLUDE_GENERAL_EXTERN_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/Utilities.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FloatingPointType.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/TypeTraitsHelper.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/GPUMutex.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/StaticAssert.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FlopsCounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaEvent.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTimer.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaAlloc.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaRefcounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMemory.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMatrix.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTypeDefs.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContextGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContext.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDevice.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaException.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMemSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrixSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaUtilities.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/Exception.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaError.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrix.hpp
    
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/PerformanceTest.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/KernelTestMethod.hpp
    
    #${COMMON_SOURCE_DIR}/external/pugixml/src/CudaFramework/pugixml.hpp
)
set(${SRC}
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaCompilerVersion.cu
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaMemSupport.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDevice.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDeviceGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContextGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContext.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaAlloc.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaUtilities.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaEvent.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaTimer.cpp
	${COMMON_SOURCE_DIR}/src/CudaFramework/General/Utilities.cpp
    
    #${COMMON_SOURCE_DIR}/external/pugixml/src/CudaFramework/pugixml.cpp
)
set(${INCLUDE_DIRS} 
    ${COMMON_SOURCE_DIR}/include/
)
endmacro(INCLUDE_GENERAL_EXTERN_CUDA)



# Macros for the JOR Prox Velocity Module
MACRO(INCLUDE_JORPROX_VELOCITY_MODULE_EXTERN_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/JORProxVel/JORProxVelocityGPUModule.cpp
    
)
set(${INC}
    ${COMMON_SOURCE_DIR}/include/CudaFramework/JORProxVel/JORProxVelocityGPUModule.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/JORProxVel/GPUBufferOffsets.hpp
)
set(${INCLUDE_DIRS}
    ${${INCLUDE_DIRS}}
    ${COMMON_SOURCE_DIR}/include/CudaFramework/JORProxVel
)
ENDMACRO(INCLUDE_JORPROX_VELOCITY_MODULE_EXTERN_CUDA)
