// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_Utilities_hpp
#define CudaFramework_General_Utilities_hpp

#include <tuple>
#include <fstream>
#include <iostream>
#include <utility>
#include <cmath>

#include <Eigen/Dense>
#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"

namespace Utilities {

std::string cutStringTillScope(std::string s);

template<typename PREC>
void fillRandom(PREC* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (PREC)RAND_MAX;
}
template<typename PREC>
void fill(PREC* data, int size, PREC value) {
    for (int i = 0; i < size; ++i)
        data[i] = value;
}

template<typename PREC>
void matrixMultiply(PREC * C, const PREC * A, const PREC * B, unsigned int MA, unsigned int NA, unsigned int MB, unsigned int NB) {
    ASSERTMSG( C!=NULL , "Reference C is NULL!");
    ASSERTMSG( A!=NULL , "Reference A is NULL!");
    ASSERTMSG( B!=NULL , "Reference B is NULL!");

    for (unsigned int i = 0; i < MA; ++i) {
        for (unsigned int j = 0; j < NB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < NA; ++k) {
                double a = A[i * NA + k];
                double b = B[k * NB + j];
                sum += a * b;
            }
            C[i * NB + j] = (PREC)sum;
        }
    }
}

template<typename PREC>
void vectorAdd(PREC * C, const PREC * A, const PREC * B, unsigned int MA) {
    ASSERTMSG( C!=NULL , "Reference C is NULL!");
    ASSERTMSG( A!=NULL , "Reference A is NULL!");
    ASSERTMSG( B!=NULL , "Reference B is NULL!");

    for (unsigned int i = 0; i < MA; ++i) {
        C[i] = A[i] + B[i];
    }
}


template<typename PREC>
bool compareArraysEach(const PREC* reference, const PREC* data, const unsigned int len, const PREC relTol, const bool abortFast = true) {
    using namespace std;

    ASSERTMSG(relTol >= 0, "relTol is smaller then 0");

    bool identical = true;
    for( unsigned int i = 0; i < len; ++i) {
        using std::abs;

        if ( abs(reference[i] - data[i]) > abs(reference[i]) * relTol) {

            identical = false;
            if(abortFast) {
                cout <<" reference["<<i<<"] = " << reference[i] <<std::endl;
                cout <<" data["<<i<<"] =" << data[i] <<std::endl;
                cout << "Compare check failed: "<<std::endl<< "abs(reference["<<i<<"] - data["<<i<<"]) =" <<abs(reference[i] - data[i]) << " > abs(reference["<<i<<"]) * relTol = "<< abs(reference[i]) * relTol <<std::endl;
                break;
            }
        };
    }

    return identical;
}

/**
* Compares two arrays and returns maxRelTol, avgRelTol, maxUlp, avgUlp,
* and bools: converged, identical, finiteReference (for NaN and Inf), finiteData (for NaN and Inf)
* converged means all relTols smaller then relTol,
* identical means all ulps smaller then maxUlpsAllowed,
*/
template<typename PREC>
std::tuple<bool,bool,bool,bool> compareArraysEachCombined( const PREC* reference,
                                                const PREC* data,
                                                const unsigned int len,
                                                const PREC relTol,
                                                const unsigned int maxUlpsAllowed,
                                                double &maxRelTol,
                                                double &avgRelTol,
                                                typename TypeWithSize<sizeof(PREC)>::UInt & maxUlp ,
                                                double &avgUlp,
                                                const bool abortFastConverged = false )
{
    bool identical = true;
    bool converged = true;
    bool isFiniteReference = true;
    bool isFiniteData = true;
    maxUlp = 0;
    avgUlp = 0;
    maxRelTol =0;
    avgRelTol = 0;
    double tol;
    typename TypeWithSize<sizeof(PREC)>::UInt value = 0;
    unsigned int i;

    PREC r,d;
    for(i = 0; i < len; ++i) {
        r = reference[i];
        d = data[i];
        using std::abs;
        using std::max;

        if (! FloatingPoint<PREC>(r).AlmostEquals(FloatingPoint<PREC>(d),maxUlpsAllowed,value)  ) {
            identical = false;
        }

        if( isFiniteReference && !std::isfinite(r)){
            isFiniteReference = false;
        }

        if( isFiniteData && !std::isfinite(d)){
            isFiniteData = false;
        }

            if(abs(r) > 0.0) {
                tol = abs(r - d) /  abs(r);
            } else {
                tol = 0;
            }
        if ( converged && (abs(r - d) > abs(r) * relTol) ) {

            converged = false;
            if(abortFastConverged) {
                i++;
                break;
            }
        }

        maxRelTol = max(maxRelTol,tol);
        maxUlp = max(maxUlp,value);

        avgRelTol += tol;
        avgUlp += value;
    }
    avgRelTol /= i;
    avgUlp /= i;

    return std::tuple<bool,bool,bool,bool>(converged,identical,isFiniteReference,isFiniteData);
}


/** This is the Gauss Seidel from the paper, this version does not respect correct ordering of already calculated values on the diagonal block */
template<typename DerivedMatrix, typename DerivedVector>
void gaussSeidelBlock(const Eigen::MatrixBase<DerivedMatrix> &G, const Eigen::MatrixBase<DerivedVector> &c, Eigen::MatrixBase<DerivedVector> &x_old,  const unsigned int nIter, const unsigned m);
/** This is the Gauss Seidel from the paper, this version  respects correct ordering of already calculated values on the diagonal block */
template<typename DerivedMatrix, typename DerivedVector>
void gaussSeidelBlockCorrect(const Eigen::MatrixBase<DerivedMatrix> &G, const Eigen::MatrixBase<DerivedVector> &c, Eigen::MatrixBase<DerivedVector> &x_old,  const unsigned int nIter, const unsigned m);

}



template<typename DerivedMatrix, typename DerivedVector>
void Utilities::gaussSeidelBlock(  const Eigen::MatrixBase<DerivedMatrix> &G,
                                   const Eigen::MatrixBase<DerivedVector> &c,
                                   Eigen::MatrixBase<DerivedVector> &x_old,
                                   const unsigned int nIter,
                                   const unsigned m) {


    typedef typename DerivedMatrix::Scalar PREC;
    Eigen::Matrix<PREC,Eigen::Dynamic,1> t;

    t = G.template triangularView< Eigen::StrictlyUpper >() * x_old;

    unsigned int n = (int)G.rows();
    unsigned int g = n / m;


    for(unsigned  int k=0; k<nIter; k++) {

        //Do for each block
        int ig;
        for(unsigned int jg=0; jg < g; jg++) {

            // (a) Do block on the diagonal
            ig = jg;

            for (unsigned int it=0; it<m; it++) {
                int i = ig*m + it;
                for (unsigned int jt=0; jt<m; jt++) {
                    int j = jg*m + jt;

                    if(i==j) {
                        x_old(i) = -(c(i) + t(i))/G(i,j);
                        t(i) = 0;
                    } else {
                        t(i) += G(i,j)*x_old(j);
                    }

                }
            }

            // (b) Do non diagonal block
            for(unsigned int it=0; it < (n-m); it++) {

                for (unsigned int jt=0; jt<m; jt++) {
                    int i = it;

                    if( it >= jg*m) {
                        i = it + m;
                    }

                    int j = jg*m + jt;

                    t(i) += G(i,j)*x_old(j);

                }

            }
        }
    }
}

template<typename DerivedMatrix, typename DerivedVector>
void Utilities::gaussSeidelBlockCorrect(  const Eigen::MatrixBase<DerivedMatrix> &G,
        const Eigen::MatrixBase<DerivedVector> &c,
        Eigen::MatrixBase<DerivedVector> &x_old,
        const unsigned int nIter,
        const unsigned m) {


    typedef typename DerivedMatrix::Scalar PREC;
    Eigen::Matrix<PREC,Eigen::Dynamic,1> t;

    t = G.template triangularView<Eigen::StrictlyUpper>()  *  x_old;

    unsigned int n = (int)G.rows();
    unsigned int g = n / m;


    for(unsigned  int k=0; k<nIter; k++) {

        //Do for each block
        int ig;
        for(unsigned int jg=0; jg < g; jg++) {

            // (a) Do block on the diagonal
            ig = jg;



            for (unsigned int jt=0; jt<m; jt++) {
                // First Calculate element on the diagonal!
                int j = jg*m + jt;
                int i = j;
                x_old(i) = -(c(i) + t(i))/G(i,j);
                t(i) = 0;

                // Propagate with the others
                for (unsigned int it=0; it<(m-1) ; it++) {
                    i = ig*m + it;
                    if(i >= j) {
                        i++;
                    }
                    t(i) += G(i,j)*x_old(j);
                }
            }

            // (b) Do non diagonal block
            for(unsigned int it=0; it < (n-m); it++) {

                for (unsigned int jt=0; jt<m; jt++) {
                    int i = it;

                    if( it >= jg*m) {
                        i = it + m;
                    }

                    int j = jg*m + jt;

                    t(i) += G(i,j)*x_old(j);

                }

            }
        }
    }
}


#endif







