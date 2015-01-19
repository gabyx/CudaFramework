// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_MatrixMultGPU_KernelsMatrixMult_cuh
#define CudaFramework_Kernels_MatrixMultGPU_KernelsMatrixMult_cuh

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

namespace matrixMultKernels {

/**
\detailed This is the simple matrix multiplication with out use of shared memory, it only operates in global memory and is slow.
Matrices are in Row-Major format and can be any size.
*/
template<typename TCudaMatrix>
__global__ void matrixMultiply_kernel(TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {

    typedef typename TCudaMatrix::PREC PREC;
    // GLOBAL MEMORY MATRIX MULT

    // Calculate  indexes
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_xStart = index_x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;



    // Each thread calculates one result entry, because our grid is two small (mostly), can also to big, calculation will still be correct,
    // The algorithm offsets then to the next same element
    // Matrices are stored in Row-Major format!
    PREC sum;
    while(index_y < C.m_M) {
        while(index_x < C.m_N) {
            //// Do matrix multiplication, one thread does A(index_y,:) * B(:,index_x)
            sum = 0;
            for(int i=0 ; i<A.m_N ; ++i) {
                sum +=  A.m_pDevice[index_y*A.m_N + i]* B.m_pDevice[ i*B.m_N + index_x];
            }
            C.m_pDevice[index_y*C.m_N + index_x] = sum;
            // =========================================================
            index_x += stride_x;
        }
        // Reset x, and move y...
        index_x = index_xStart;
        index_y += stride_y;
    }

}


/** This is dynamically allocated shared memory which is specified by the kernel call! */
extern __shared__  float shared_arrayDynAlloc[];
/**
\detailed This is the Shared Method.
Matrices are in Row-Major format and can be any size. This function has dim-safe access functions, which makes it slow!
This shared matrix multiplication is fast, but suffers from not useage of the whole capacity, resource spilling.
The block size in which Matrix C is split can be specified in this function.
*/
template<typename TCudaMatrix>
__global__ void matrixMultiplyShared_kernel( TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {
    typedef typename TCudaMatrix::PREC PREC;
    // SHARED MEMORY MATRIX MULT

    // Grid size
    int gDimX = gridDim.x;
    int gDimY = gridDim.y;

    // Block size need to be SYMETRIC!!! checked in kernel wrapper
    int bDim = blockDim.x;

    // Block row and column
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Number of Sum SubMatrices needed;
    int nSumBlocks = (A.m_N + bDim-1) / bDim;
    // e.g (51 + 10 - 1) / 10 = 6  till (60 + 10 - 1) / 10 = 6

    // Number of Blocks needed to fully fill C
    int maxBlocksCX = (C.m_N + bDim-1) / bDim;
    int maxBlocksCY = (C.m_M + bDim-1) / bDim;

    dim3 currentCBlockIdx(bx,by);

    // Shifting Blocks, only done beacuse it may happen that the grid is two small for the matrix!
    while( currentCBlockIdx.y < maxBlocksCY) {
        while( currentCBlockIdx.x < maxBlocksCX ) {

            // This matrices are shared in each block [bDimX,bDimY]! =======================================
            // Dynamic Size

            int size = (bDim*bDim);
            PREC* Asub_sh = (PREC *) shared_arrayDynAlloc;
            PREC* Bsub_sh = (PREC *) &shared_arrayDynAlloc[size*sizeof(PREC) / sizeof(float)];

            // Fixed Size TEST (blockDim = 16!!!)
            //__shared__ PREC Asub_sh[16*16];
            //__shared__ PREC Bsub_sh[16*16];
            // =============================================================================================

            // Calculate Csub for this current Block Matrix ===========================================================================
            // First load the i-th Matrix Asub_sh {1...nSumBlocks}  and Bsub_s {1...nSumBlocks} into shared memory
            // Each thread in a Block writes one element to the shared matrix Asub_sh, Bsub_sh
            // We iterate over all Asub,Bsub matrices always calculating sums to sum up to the correct value
            PREC currentCBlock_value_txty = 0; // Sum value for the current tx,ty in the matrix Csub (currentCBlockIdx)


            for(int currentSumBlockIdx = 0; currentSumBlockIdx < nSumBlocks ; currentSumBlockIdx++) {

                // Load i-th SubSum Matrix
                Asub_sh[ty* bDim + tx] = getElementOrZero(A,	ty + currentCBlockIdx.y * bDim, tx + currentSumBlockIdx * bDim);
                Bsub_sh[ty* bDim + tx] = getElementOrZero(B,	ty + currentSumBlockIdx * bDim,	tx + currentCBlockIdx.x * bDim);
                __syncthreads();

                // Sum the currentSumBlockIdx Matrix (corresponding row and column)
                for( int k = 0; k < bDim; k++) {
                    currentCBlock_value_txty += Asub_sh[ty*bDim + k] * Bsub_sh[k*bDim + tx];
                }

                // Add temporal value back on the C matrix, do this safe, because the currentCBlock may be out of the reach in C...
                setElementDimSafe(C, ty + currentCBlockIdx.y * bDim, tx + currentCBlockIdx.x * bDim, currentCBlock_value_txty);

                // Sync threads befor one thread loads a new matrix!
                __syncthreads();

            }
            //=========================================================================================================================

            // Jump to next block column
            currentCBlockIdx.x += gDimX;
        }

        // Jump to next row
        currentCBlockIdx.x = bx;
        currentCBlockIdx.y += gDimY;
    }

}


/** Compile define for matrixMultiplySharedFixed_kernelWrap */
#define mMSF_BLOCKSIZE 32
/**
\detailed This is the Shared Method.
Matrices are in Row-Major format and can be any size. This function has dim-safe access functions, which makes it slow!
This shared matrix multiplication is fast, but suffers from not useage of the whole capacity, resource spilling.
The Matrix C is split into mMSF_BLOCKSIZE x mMSF_BLOCKSIZE subblocks.
\param mMSF_BLOCKSIZE  This define is used to set the symetric Blocksize at compile time.
*/
template<typename TCudaMatrix>
__global__ void matrixMultiplySharedFixed_kernel( TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {
    typedef typename TCudaMatrix::PREC PREC;
    // SHARED MEMORY MATRIX MULT

    // Grid size
    int gDimX = gridDim.x;
    int gDimY = gridDim.y;

    // Block size need to be SYMETRIC!!! checked in kernel wrapper
    int bDim = mMSF_BLOCKSIZE;

    // Block row and column
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Number of Sum SubMatrices needed;
    int nSumBlocks = (A.m_N + bDim-1) / bDim;
    // e.g (51 + 10 - 1) / 10 = 6  till (60 + 10 - 1) / 10 = 6

    // Number of Blocks needed to fully fill C
    int maxBlocksCX = (C.m_N + bDim-1) / bDim;
    int maxBlocksCY = (C.m_M + bDim-1) / bDim;

    dim3 currentCBlockIdx(bx,by);

    // Shifting Blocks, only done beacuse it may happen that the grid is two small for the matrix!
    while( currentCBlockIdx.y < maxBlocksCY) {
        while( currentCBlockIdx.x < maxBlocksCX ) {

            // This matrices are shared in each block [bDimX,bDimY]! =======================================
            // Dynamic Size

            /*int size = (bDim*bDim);
            PREC* Asub_sh = (PREC *) shared_arrayDynAlloc;
            PREC* Bsub_sh = (PREC *) &shared_arrayDynAlloc[size];*/

            // Fixed Size TEST (blockDim = 16!!!)
            __shared__ PREC Asub_sh[mMSF_BLOCKSIZE*mMSF_BLOCKSIZE];
            __shared__ PREC Bsub_sh[mMSF_BLOCKSIZE*mMSF_BLOCKSIZE];
            // =============================================================================================

            // Calculate Csub for this current Block Matrix ===========================================================================
            // First load the i-th Matrix Asub_sh {1...nSumBlocks}  and Bsub_s {1...nSumBlocks} into shared memory
            // Each thread in a Block writes one element to the shared matrix Asub_sh, Bsub_sh
            // We iterate over all Asub,Bsub matrices always calculating sums to sum up to the correct value
            PREC currentCBlock_value_txty = 0; // Sum value for the current tx,ty in the matrix Csub (currentCBlockIdx)


            for(int currentSumBlockIdx = 0; currentSumBlockIdx < nSumBlocks ; currentSumBlockIdx++) {

                // Load i-th SubSum Matrix
                Asub_sh[ty* bDim + tx] = getElementOrZero(A,	ty + currentCBlockIdx.y * bDim, tx + currentSumBlockIdx * bDim);
                Bsub_sh[ty* bDim + tx] = getElementOrZero(B,	ty + currentSumBlockIdx * bDim,	tx + currentCBlockIdx.x * bDim);
                __syncthreads();

                // Sum the currentSumBlockIdx Matrix (corresponding row and column)
#pragma unroll
                for( int k = 0; k < mMSF_BLOCKSIZE; k++) {
                    currentCBlock_value_txty += Asub_sh[ty*bDim + k] * Bsub_sh[k*bDim + tx];
                }

                // Add temporal value back on the C matrix, do this safe, because the currentCBlock may be out of the reach in C...
                setElementDimSafe(C, ty + currentCBlockIdx.y * bDim, tx + currentCBlockIdx.x * bDim, currentCBlock_value_txty);

                // Sync threads befor one thread loads a new matrix!
                __syncthreads();

            }
            //=========================================================================================================================

            // Jump to next block column
            currentCBlockIdx.x += gDimX;
        }

        // Jump to next row
        currentCBlockIdx.x = bx;
        currentCBlockIdx.y += gDimY;
    }

}


/**
\detailed This is the Row-Major Method.
Matrices are in Row-Major format!
It splites C into Blocks of 256x16 in which 256 (16x16) threads are assigned.
Each thread loads one element to Asub_sh (16x16).
This method works not as expected, and is slow compared to the ...Col_kernel which uses coalesced memory acces because each thread loads
Bsub and all are accessed in linear memory order, which is collected from the GPU.
*/
template<typename TCudaMatrix>
__global__ void matrixMultiplySharedFixedLargeRow_kernel( TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {
    typedef typename TCudaMatrix::PREC PREC;
    // SHARED MEMORY MATRIX MULT

    // Grid size
    int gDimX = gridDim.x;
    int gDimY = gridDim.y;

    // Block size need to be 16x16 threads/block!!! checked in kernel wrapper

    // Block row and column
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tlinear = tx+16*ty;

    // Number of Sum SubMatrices needed;
    int nSumBlocks = (A.m_N + 16-1) / 16;
    // e.g (51 + 10 - 1) / 10 = 6  till (60 + 10 - 1) / 10 = 6

    // Number of Blocks needed to fully fill C
    int maxBlocksCX = (C.m_N + (16-1)) / (16);
    int maxBlocksCY = (C.m_M + (256-1)) / (256);

    dim3 currentCBlockIdx(bx,by);

    // Shifting Blocks of 256x16 in C
    while( currentCBlockIdx.y < maxBlocksCY) {
        while( currentCBlockIdx.x < maxBlocksCX ) {

            // This matrices are shared in each block [bDimX,bDimY]! =======================================
            // Dynamic Size

            // Fixed Size TEST (blockDim = 16!!!)
            __shared__ PREC Bsub_sh[16][16]; // row major format!
            // =============================================================================================

            // Calculate Csub for this current Block Matrix ===========================================================================
            // First load the i-th Matrix Asub_sh {1...nSumBlocks}  and Bsub_s {1...nSumBlocks} into shared memory
            // Each thread in a Block writes one element to the shared matrix Asub_sh, Bsub_sh
            // We iterate over all Asub,Bsub matrices always calculating sums to sum up to the correct value

            PREC Crow[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


            for(int currentSumBlockIdx = 0; currentSumBlockIdx < nSumBlocks ; currentSumBlockIdx++) {

                // Load i-th SubSum Matrix, Row Major
                Bsub_sh[ty][tx] = getElementOrZero(B,	ty + currentSumBlockIdx * 16, tx + currentCBlockIdx.x * 16);
                __syncthreads();


                // Sum the Row for the current thread


#pragma unroll
                for( int kColElemA = 0; kColElemA < 16; kColElemA++) {

                    PREC Asub = getElementOrZero(A, tlinear + currentCBlockIdx.y * 256, kColElemA + currentSumBlockIdx * 16);

#pragma unroll
                    for(int iColB=0; iColB<16; iColB++) {
                        Crow[iColB] += Asub * Bsub_sh[kColElemA][iColB];
                    }

                }

                // Add temporal Crow back to C, each thread writes 16 values
#pragma unroll
                for(int i=0; i<16; i++) {
                    setElementDimSafe(C, tlinear + currentCBlockIdx.y * 256, i + currentCBlockIdx.x * 16, Crow[i]);
                }

                // Sync threads befor one thread loads a new matrix!
                __syncthreads();

            }
            //=========================================================================================================================

            // Jump to next block column
            currentCBlockIdx.x += gDimX;
        }

        // Jump to next row
        currentCBlockIdx.x = bx;
        currentCBlockIdx.y += gDimY;
    }

}


/**
\detailed This is the Col-Major Method.
Matrices are in Row-Major format and can be any size. This function has dim-safe access functions, which makes it slow!
Each thread loads one element to Bsub_sh (16x16).
It splites C into Blocks of 16x512 in which 512 (16x16) threads are assigned. This call uses full register/blocks for Tesla C2050
This method is fast and the same as the matrixMultiplySharedFixedLargeBase_kernelWrap().
*/
template<typename TCudaMatrix>
__global__ void matrixMultiplySharedFixedLargeCol_kernel( TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {
    typedef typename TCudaMatrix::PREC PREC;
    // SHARED MEMORY MATRIX MULT

    // Grid size
    int gDimX = gridDim.x;
    int gDimY = gridDim.y;

    // Block size need to be 16x16 threads/block!!! checked in kernel wrapper

    // Block row and column
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tlinear = tx + 16 * ty;

    // Number of Sum SubMatrices needed;
    int nSumBlocks = (A.m_N + 16-1) / 16;
    // e.g (51 + 10 - 1) / 10 = 6  till (60 + 10 - 1) / 10 = 6

    // Number of Blocks needed to fully fill C
    int maxBlocksCX = (C.m_N + (256-1)) / (256);
    int maxBlocksCY = (C.m_M + (16-1)) / (16);

    dim3 currentCBlockIdx(bx,by);

    // Shifting Blocks of 256x16 in C
    while( currentCBlockIdx.y < maxBlocksCY) {
        while( currentCBlockIdx.x < maxBlocksCX ) {

            // This matrices are shared in each block [bDimX,bDimY]! =======================================
            // Dynamic Size

            // Fixed Size TEST (blockDim = 16!!!)
            __shared__ PREC Asub_sh[16][18]; // row major format!, Asub_sh[i][j] => Asub_sh[i*17 + j]
            // we add a padding row, to avoid bank conflicts
            // =============================================================================================

            // Calculate Csub for this current Block Matrix ===========================================================================
            // First load the i-th Matrix Asub_sh {1...nSumBlocks}  and Bsub_s {1...nSumBlocks} into shared memory
            // Each thread in a Block writes one element to the shared matrix Asub_sh, Bsub_sh
            // We iterate over all Asub,Bsub matrices always calculating sums to sum up to the correct value

            PREC Crow[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            for(int currentSumBlockIdx = 0; currentSumBlockIdx < nSumBlocks ; currentSumBlockIdx++) {

                // Load i-th SubSum Matrix, Column Major
                Asub_sh[tx][ty] = getElementOrZero(A,	ty + currentCBlockIdx.y * 16, tx + currentSumBlockIdx * 16); // Asub_sh[tx*17+ty]
                __syncthreads();

                // Sum the Row for the current thread

#pragma unroll
                for( int k = 0; k < 16; k++) {

                    PREC Bsub = getElementOrZero(B, k + currentSumBlockIdx * 16, tlinear + currentCBlockIdx.x * 256);

#pragma unroll
                    for(int i=0; i<16; i++) {
                        Crow[i] += Bsub * Asub_sh[k][i];
                    }
                    //compm16(Bsub, &Asub_sh[k][0], Crow);
                }
                // Sync threads befor one thread loads a new matrix!
                __syncthreads();

            }

            // Add temporal Crow back to C, each thread writes 16 values
#pragma unroll
            for(int i=0; i < 16 ; i++) {
                setElementDimSafe(C, i + currentCBlockIdx.y * 16, tlinear + currentCBlockIdx.x * 256, Crow[i]);
            }
            //=========================================================================================================================

            // Jump to next block column
            currentCBlockIdx.x += gDimX;
        }

        // Jump to next row
        currentCBlockIdx.x = bx;
        currentCBlockIdx.y += gDimY;
    }

}

/**
\detailed This is the Col-Major Method.
Matrices are in Row-Major format and can be a multiple of 16x256 in each dimension. This function has NOT dim-safe access functions, which makes it fast!
This method is optimized!
*/
template<typename TCudaMatrix>
__global__ void matrixMultiplySharedFixedLargeColOptimized_kernel(TCudaMatrix C, TCudaMatrix A, TCudaMatrix B) {
    typedef typename TCudaMatrix::PREC PREC;
    // SHARED MEMORY MATRIX MULT

    // Grid size
    //int gDimX = gridDim.x;
    //int gDimY = gridDim.y;

    // Block size need to be 16x16 threads/block!!! checked in kernel wrapper

    // Block row and column
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tlinear = tx+16*ty;

    // Number of Sum SubMatrices needed;
    int nSumBlocks = (A.m_N + 16-1) / 16;
    // e.g (51 + 10 - 1) / 10 = 6  till (60 + 10 - 1) / 10 = 6

    // Number of Blocks needed to fully fill C
    //int maxBlocksCX = (C.m_N + (256-1)) / (256);
    //int maxBlocksCY = (C.m_M + (16-1)) / (16);

    dim3 currentCBlockIdx(bx,by);

    // Shifting Blocks of 16x256 in C
    PREC *a,*b,*c;

    /*while( currentCBlockIdx.y < maxBlocksCY){
    while( currentCBlockIdx.x < maxBlocksCX ){*/

    //offset to the right element in A, to copy out to Asub
    a = &A.m_pDevice[(currentCBlockIdx.y * 16) * A.m_N + (ty*A.m_N + tx)];
    // offset to the right column in B (first sum block)
    b = &B.m_pDevice[tlinear + currentCBlockIdx.x * 256];
    // offset to the right column in C
    c = &C.m_pDevice[(currentCBlockIdx.y * 16) * C.m_N + tlinear + currentCBlockIdx.x * 256];

    // This matrices are shared in each block [bDimX,bDimY]! =======================================
    __shared__ PREC Asub_sh[16][17]; // row major format!, Asub_sh[i][j] => Asub_sh[i*17 + j]
    // we add a padding row, to avoid bank conflicts
    // =============================================================================================

    // Calculate Csub for this current Block Matrix ===========================================================================
    // First load the i-th Matrix Asub_sh {1...nSumBlocks}  and Bsub_s {1...nSumBlocks} into shared memory
    // Each thread in a Block writes one element to the shared matrix Asub_sh, Bsub_sh
    // We iterate over all Asub,Bsub matrices always calculating sums to sum up to the correct value

    PREC Crow[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


    for(int currentSumBlockIdx = 0; currentSumBlockIdx < nSumBlocks ; currentSumBlockIdx++) {

        // Load i-th SubSum Matrix, Column Major

        Asub_sh[tx][ty] = a[0]; // Asub_sh[tx*17+ty]
        __syncthreads();


        // Sum the Row for the current thread
#pragma unroll
        for( int k = 0; k < 16; k++) {

            PREC Bsub = b[ k*B.m_N ]; // if here the size is know before compile time, it is 100Gflops faster than, if the B.m_N is dynamic, crazy :-)

#pragma unroll
            for(int i=0; i<16; i++) {
                Crow[i] += Bsub * Asub_sh[k][i];
            }
            //comp16(Bsub, &Asub_sh[k][0], Crow);
        }
        a +=16;
        b +=16*B.m_N;

        // Sync threads befor one thread loads a new matrix!
        __syncthreads();
    }

    // Add temporal Crow back to C, each thread writes 16 values
#pragma unroll
    for(int i=0; i<16; i++) {
        *c = Crow[i];
        c += C.m_N;
    }

    //=========================================================================================================================

    //		// Jump to next block column
    //		currentCBlockIdx.x += gDimX;
    //	}

    //	// Jump to next row
    //	currentCBlockIdx.x = bx;
    //	currentCBlockIdx.y += gDimY;
    //}

}


/** Compute 16 MADs (Multiply-and-Add). One MAD for each output value. */
template<typename PREC>
__device__ void comp16(PREC a, PREC *b, PREC *c) {
    c[0] += a*b[0];
    c[1] += a*b[1];
    c[2] += a*b[2];
    c[3] += a*b[3];
    c[4] += a*b[4];
    c[5] += a*b[5];
    c[6] += a*b[6];
    c[7] += a*b[7];
    c[8] += a*b[8];
    c[9] += a*b[9];
    c[10] += a*b[10];
    c[11] += a*b[11];
    c[12] += a*b[12];
    c[13] += a*b[13];
    c[14] += a*b[14];
    c[15] += a*b[15];
}
/** This matrix multiplications work with multiple of 256 square matrices, from http://coachk.cs.ucf.edu/courses/CDA6938/s08/matrixmul_hongliang.cu */
#define MATRIX_WIDTH 4096
template<typename PREC>
__global__ void matrix_large_tile_optimized(PREC *A, PREC *B, PREC *C) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x * 256;
    int by = blockIdx.y * 16;

    A += by * MATRIX_WIDTH + tx + ty * MATRIX_WIDTH;
    B += bx + tx + ty * 16;
    C += by * MATRIX_WIDTH + bx + tx + ty * 16;

    // a is the register used to prefetch a value from A
    PREC a = A[0];

    // b is the array to prefetch 4 values a time from B
    PREC b[4] = {B[0], B[MATRIX_WIDTH], B[2*MATRIX_WIDTH], B[3*MATRIX_WIDTH]};

    PREC *Alast = A + MATRIX_WIDTH;

    A += 16;

    __shared__ PREC ashare[16][17];

    PREC c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // bb is used to buffer prefetched values in b
    PREC bb[4];
    do {
        ashare[tx][ty] = a;
        __syncthreads();

        // prefetch A[0]
        a = A[0];

        bb[0] = b[0];
        bb[1] = b[1];
        bb[2] = b[2];
        bb[3] = b[3];
        b[0] = B[4 * MATRIX_WIDTH];
        b[1] = B[5 * MATRIX_WIDTH];
        b[2] = B[6 * MATRIX_WIDTH];
        b[3] = B[7 * MATRIX_WIDTH];
        for (int i = 0; i < 4; i ++)
            comp16(bb[i], &ashare[i][0], c);

        bb[0] = b[0];
        bb[1] = b[1];
        bb[2] = b[2];
        bb[3] = b[3];
        b[0] = B[8 * MATRIX_WIDTH];
        b[1] = B[9 * MATRIX_WIDTH];
        b[2] = B[10 * MATRIX_WIDTH];
        b[3] = B[11 * MATRIX_WIDTH];
        for (int i = 0; i < 4; i ++)
            comp16(bb[i], &ashare[i + 4][0], c);

        bb[0] = b[0];
        bb[1] = b[1];
        bb[2] = b[2];
        bb[3] = b[3];
        b[0] = B[12 * MATRIX_WIDTH];
        b[1] = B[13 * MATRIX_WIDTH];
        b[2] = B[14 * MATRIX_WIDTH];
        b[3] = B[15 * MATRIX_WIDTH];
        for (int i = 0; i < 4; i ++)
            comp16(bb[i], &ashare[i + 8][0], c);

        bb[0] = b[0];
        bb[1] = b[1];
        bb[2] = b[2];
        bb[3] = b[3];

        A += 16;
        B += 16 * MATRIX_WIDTH;
        b[0] = B[0 * MATRIX_WIDTH];
        b[1] = B[1 * MATRIX_WIDTH];
        b[2] = B[2 * MATRIX_WIDTH];
        b[3] = B[3 * MATRIX_WIDTH];
        for (int i = 0; i < 4; i ++)
            comp16(bb[i], &ashare[i + 12][0], c);
        __syncthreads();
    } while( A < Alast );

    // to prevent overflow of prefetching instructions in the loop, we
    // calculate the last iteration here without prefetches:

    ashare[tx][ty] = a;
    __syncthreads();

    bb[0] = b[0];
    bb[1] = b[1];
    bb[2] = b[2];
    bb[3] = b[3];
    b[0] = B[4 * MATRIX_WIDTH];
    b[1] = B[5 * MATRIX_WIDTH];
    b[2] = B[6 * MATRIX_WIDTH];
    b[3] = B[7 * MATRIX_WIDTH];
    for (int i = 0; i < 4; i ++)
        comp16(bb[i], &ashare[i][0], c);
    bb[0] = b[0];
    bb[1] = b[1];
    bb[2] = b[2];
    bb[3] = b[3];
    b[0] = B[8 * MATRIX_WIDTH];
    b[1] = B[9 * MATRIX_WIDTH];
    b[2] = B[10 * MATRIX_WIDTH];
    b[3] = B[11 * MATRIX_WIDTH];
    for (int i = 0; i < 4; i ++)
        comp16(bb[i], &ashare[i + 4][0], c);
    bb[0] = b[0];
    bb[1] = b[1];
    bb[2] = b[2];
    bb[3] = b[3];
    b[0] = B[12 * MATRIX_WIDTH];
    b[1] = B[13 * MATRIX_WIDTH];
    b[2] = B[14 * MATRIX_WIDTH];
    b[3] = B[15 * MATRIX_WIDTH];
    for (int i = 0; i < 4; i ++)
        comp16(bb[i], &ashare[i + 8][0], c);
    for (int i = 0; i < 4; i ++)
        comp16(b[i], &ashare[i + 12][0], c);

    // output
    for(int i = 0; i < 16; i++, C += MATRIX_WIDTH )
        C[0] = c[i];
}
template<typename PREC>
__global__ void matrix_large_tile_base(PREC *A, PREC *B, PREC *C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x * 256;
    int by = blockIdx.y * 16;

    A += by * MATRIX_WIDTH + tx + ty * MATRIX_WIDTH; // offset to first submatrix A
    B += bx + (tx + ty * 16);  //offset to first B
    C += by * MATRIX_WIDTH + bx + (tx + ty * 16); //offset to first C

    // Subtile of matrix A is stored in shared memory.
    // A padding row is added to remove bank conflicts.
    __shared__ PREC ashare[16][17];

    // 16 output values.
    PREC c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // A register to store values from the subtile of matrix B.
    PREC b;

    // for each subtile in A and B
    for (int i = 0; i < MATRIX_WIDTH/16; i++) {
        // read in the subtile of A
        ashare[tx][ty] = A[0]; // each thread writes the first value
        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; k++) {
            // read in one value of subtile of B, which is shared by
            // all 16 output values.
            b = B[k * MATRIX_WIDTH];

            comp16(b, &ashare[k][0], c);
        }

        // move to next subtile
        A += 16;
        B += 16 * MATRIX_WIDTH;
        __syncthreads();
    };

    for(int i = 0; i < 16; i++) {
        C[0] = c[i];
        C += MATRIX_WIDTH;
    }
}



/** Get a matrix element */
template<typename TCudaMatrix>
__forceinline__ __device__ typename TCudaMatrix::PREC getElement(const TCudaMatrix A, int row, int col) {
    typedef typename TCudaMatrix::PREC PREC;
    return Elem_RowM(A,row,col);
}
/** Get a matrix element */
template<typename TCudaMatrix>
__forceinline__ __device__  typename TCudaMatrix::PREC getElementOrZero(const TCudaMatrix A, int row, int col) {
typedef typename TCudaMatrix::PREC PREC;
    PREC ret = 0;
    if( row < A.m_M && col < A.m_N) {
        ret = Elem_RowM(A,row,col);
    }
    return ret; // A.m_pDevice[row * A.outerStride + col];
}

/** Set a matrix element */
template<typename TCudaMatrix>
__forceinline__ __device__ void setElement(TCudaMatrix A, int row, int col, typename TCudaMatrix::PREC value) {
    typedef typename TCudaMatrix::PREC PREC;
    Elem_RowM(A,row,col)= value;
}
/** Set a matrix element  */
template<typename TCudaMatrix>
__forceinline__ __device__ void setElementDimSafe(TCudaMatrix A, int row, int col, typename TCudaMatrix::PREC value) {
    typedef typename TCudaMatrix::PREC PREC;
    if( row < A.m_M && col < A.m_N) {
        Elem_RowM(A,row,col) = value;
    }
}

/**
* Get the sizeX x sizeY sub-matrix Asub of A that is
* located col sub-matrices to the right and row sub-matrices down
* from the upper-left corner of A
*/
template<typename TCudaMatrix>
__device__ TCudaMatrix getSubMatrix(TCudaMatrix A, int row, int col, int sizeX, int sizeY) {
    TCudaMatrix Asub;
    Asub.m_N = sizeX;
    Asub.m_M = sizeY;
    Asub.outerStride = A.outerStride;
    Asub.m_pDevice = &A.m_pDevice[A.outerStride * sizeY * row + sizeX * col];
    return Asub;
}


}
#endif
