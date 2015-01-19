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

#ifndef CudaFramework_CudaModern_CudaMatrix_hpp
#define CudaFramework_CudaModern_CudaMatrix_hpp


/** General Device Macro for row-major matrices */
#define PtrElem_RowM(_A,_row,_col)                          (        (PREC*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col)       )
#define Elem_RowM(_A,_row,_col)                             (     *( (PREC*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col) )     )
#define PtrRowOffset_RowM(_A_ptr,_rows,_outerStrideBytes)   (        (PREC*)((char*)(_A_ptr)  + (_rows) * (_outerStrideBytes))                 )


/** General Device Macro for col-major matrices */
#define PtrElem_ColM(_A,_row,_col)                                PtrElem_RowM(_A,_col,_row)
#define Elem_ColM(_A,_row,_col)                                   Elem_RowM(_A,_col,_row)
#define PtrColOffset_ColM(_A_ptr,_cols,_outerStrideBytes)         PtrRowOffset_RowM(_A_ptr,_cols,_outerStrideBytes)

/** Super General Device Macro to choose the right Offset depending on the Flags */
#define PtrElem_Row(_AType,_A,_row,_col) ( (_AType::Flags  &  CudaMatrixFlags::RowMajorBit)? ( PtrElem_RowM(_A,_row,_col) ) : ( PtrElem_ColM(_A,_row,_col) ) )


namespace utilCuda{


    namespace CudaMatrixFlags {
        const unsigned int RowMajorBit = 0x1;     // Defines the storage order!

        // Do not use AlignedBit as a flag for CudaMatrix
        // const unsigned int AlignedBit  = 0x1 << 1; // Makes sure the Matrix on the device is aligned for coalesced access!
        // const unsigned int RowMajorAligned = AlignedBit  | RowMajorBit;
        // const unsigned int RowMajorNotAligned = RowMajorBit;
        //
        // const unsigned int ColMajorAligned = AlignedBit;
        // const unsigned int ColMajorNotAligned = 0;

        const unsigned int RowMajor = RowMajorBit;
        const unsigned int ColMajor = 0;

        template<unsigned int Flag>
        struct isRowMajor{
            static const bool value = (Flag & RowMajorBit)? true : false;
        };

        //template<unsigned int Flag>
        //struct isAligned{
        //    static const bool value = (Flag & AlignedBit)?  true : false;
        //};

    };


    /**
    *   This matrix struct serves as a pass by value struct for the parameters to the kernel!
    *	If we rather would not like to pass by value to the kernel (because of several iterations) we are
    *   free to copy the parameters to constant memory once and pass a constant_memory_device ptr to the kernel
    *   The kernel then needs to unpack all parameters from constant memory to obtain the values in this struct again.
    *   This is complicated but would be a nice try to see what performance gain we achieve!
    */


    template<typename TPREC, unsigned int InputFlags = CudaMatrixFlags::ColMajor >
    struct CudaMatrix{
            typedef TPREC PREC;
            typedef unsigned int SizeType;

            CudaMatrix(): m_M(0),m_N(0),m_outerStrideBytes(0),m_pDevice(0)
            {}

            CudaMatrix(SizeType M,SizeType N,SizeType outerStrideBytes):
                m_M(M),m_N(N),m_outerStrideBytes(outerStrideBytes),m_pDevice(0)
            {}

            SizeType m_M; //witdh
            SizeType m_N; //height
            SizeType m_outerStrideBytes;
            PREC* m_pDevice;


            //Compile-Time Enums (not stored at runtime on the GPU)
            enum {
                Flags = InputFlags
            };

    };

};

#endif // CudaMatrix_hpp
