/******************************************************************************
* Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the NVIDIA CORPORATION nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* 
*  Source code modified and extended from moderngpu.com
******************************************************************************/

#ifndef CudaFramework_CudaModern_CudaPrint_hpp
#define CudaFramework_CudaModern_CudaPrint_hpp

#include <cstdarg>
#include <iterator>
#include <algorithm>
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMemory.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMatrix.hpp"

#define MACRO_DIV_UP(x, y) (((x) + (y) - 1) / (y))

namespace utilCuda{

    namespace internal{

        //
        std::string stringprintf(const char* format, ...);

        /** rowNumber: 0 = no row number, 1= element count at this row, 2= row number as in matrix notation
        *
        */
        template<int rowNumber, typename T, typename Op>
        std::string formatArray(const T* data, size_t count, Op op, int numCols) {
            std::string s;
            size_t numRows = MACRO_DIV_UP(count, numCols);
            for(size_t row(0); row < numRows; ++row) {
                size_t left = row * numCols;

                switch(rowNumber){
                    case 1:
                        s.append(stringprintf("%5d:  ", left)); break;
                    case 2:
                        s.append(stringprintf("%5d:  ", row)); break;
                }


                for(size_t col(left); col < std::min(left + numCols, count); ++col) {
                    s.append(op(col, data[col]));
                    s.push_back(' ');
                }
                s.push_back('\n');
            }
            return s;
        }

        template<int rowNumber, typename T, typename Op>
        std::string formatArray(const std::vector<T>& data, Op op, int numCols) {
            return formatArray<rowNumber>(&data[0], (int)data.size(), op, numCols);
        }


        template<typename T, typename Op>
        std::string formatArray(const CudaDeviceMem<T>& mem, unsigned int count, Op op, unsigned int numCols) {
            std::vector<T> hostMem;
            ASSERTCHECK_CUDA(mem.toHost(hostMem, count));
            return formatArray<1>(hostMem, op, numCols);
        }

        template<typename T, bool RowMajor, typename Op>
        std::string formatArray(const CudaDeviceMatrix<T, RowMajor>& mem, Op op) {
            auto matrix = mem.get();
            std::vector<T> hostMem(matrix.m_M*matrix.m_N);
            ASSERTCHECK_CUDA(mem.toHost(&hostMem[0], matrix.m_M, matrix.m_N));
            return formatArray<2>(hostMem, op, matrix.m_N);
        }



        struct FormatFunctorPrintf {
            const char* format;
            FormatFunctorPrintf(const char* f) : format(f) { }

            template<typename T>
            inline std::string operator()(int index, T x) const {
                return stringprintf(format, x);
            }
        };


    };

    template<typename T>
    void printArray(const CudaDeviceMem<T>& mem, unsigned int count, const char* format, unsigned int numCols)    {
        std::string s = internal::formatArray(mem, count, internal::FormatFunctorPrintf(format), numCols);
        printf("%s", s.c_str());
    }

    template<typename T>
    void printArray(const CudaDeviceMem<T>& mem, const char* format, unsigned int numCols) {
         printArray(mem,mem.size(),format,numCols);
    }

    template<typename T, bool RowMajor>
    void printArray(const CudaDeviceMatrix<T,RowMajor> & matrix, const char* format) {
         std::string s = internal::formatArray(matrix, internal::FormatFunctorPrintf(format));
         printf("%s", s.c_str());
    }


};


#undef MACRO_DIV_UP

#endif // CudaPrint_hpp
