/*
 *  GRSF/Dynamics/General/MyMatrixDefs.hpp
 *
 *  Created by Gabriel NÃ¼tzi on 21.03.10.
 *  Copyright 2010 ETH. All rights reserved.
 *
 */

#ifndef CudaFramework_General_MyMatrixTypeDefs_hpp
#define CudaFramework_General_MyMatrixTypeDefs_hpp

//#define EIGEN_DONT_VECTORIZE
//#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include <vector>
#include <map>
#include <unordered_map>
//#include <Eigen/StdVector>


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

#include <Eigen/Eigenvalues>

// ================================================================================================
/** @brief This
*	These are some small matrix definitions.
*/
template<typename TPREC>
struct MyMatrix {
    using PREC = TPREC;
    //Static assigned Matrices
    using Matrix44 = Eigen::Matrix<PREC, 4, 4>;
    using Matrix43 = Eigen::Matrix<PREC, 4, 3>;
    using Matrix34 = Eigen::Matrix<PREC, 3, 4>;
    using Matrix33 = Eigen::Matrix<PREC, 3, 3>;
    using Matrix22 = Eigen::Matrix<PREC, 2, 2>;
    using Vector3 = Eigen::Matrix<PREC, 3, 1>;
    using Vector2 = Eigen::Matrix<PREC, 2, 1>;

    using Quaternion = Eigen::Quaternion<PREC>;
    using AngleAxis = Eigen::AngleAxis<PREC>;

    using Vector4 = Eigen::Matrix<PREC, 4, 1>;
    using Vector6 = Eigen::Matrix<PREC, 6, 1>;
    using VectorDyn = Eigen::Matrix<PREC, Eigen::Dynamic , 1 >                   ;


    using MatrixDynDyn = Eigen::Matrix<PREC, Eigen::Dynamic , Eigen::Dynamic >      ;
    using MatrixDiagDyn = Eigen::DiagonalMatrix<PREC, Eigen::Dynamic >               ;
    using MatrixDynDynRow = Eigen::Matrix<PREC, Eigen::Dynamic , Eigen::Dynamic, Eigen::RowMajor>;

    template<unsigned int M>
    using MatrixStatDyn = Eigen::Matrix<PREC, M, Eigen::Dynamic >;
    template<unsigned int N>
    using MatrixDynStat = Eigen::Matrix<PREC, Eigen::Dynamic, N >;

    using AffineTrafo = Eigen::Transform<PREC,3,Eigen::TransformTraits::Affine>;
    using AffineTrafo2d = Eigen::Transform<PREC,2,Eigen::TransformTraits::Affine>;

    using MatrixSparse = Eigen::SparseMatrix<PREC>   ;       // declares a column-major sparse matrix of type PREC
    using MatrixSparseTriplet = Eigen::Triplet<PREC>        ;


    template<typename Derived> using MatrixBase = Eigen::MatrixBase<Derived>;
    template<typename Derived> using ArrayBase  = Eigen::ArrayBase<Derived>;

    template<typename Derived>                 using MatrixBDynDyn  = Eigen::Block<Derived>;
    template<typename Derived, unsigned int M> using MatrixBStatDyn = Eigen::Block<Derived,M>;
    template<typename Derived, unsigned int N> using MatrixBDynStat = Eigen::Block<Derived,Eigen::Dynamic,N>;

    template<typename EigenType> using MatrixRef = Eigen::Ref<EigenType>;

    template<typename EigenType> using MatrixMap = Eigen::Map<EigenType>;




    // Sepcial STL map where the type is 16byte aligned
    template<typename Key, typename Type, typename Comp = std::less<Key> >
    using StdMapAligned = std::map<Key, Type, Comp, Eigen::aligned_allocator<std::pair<const Key, Type> > >;

    // Sepcial STL map where the type is 16byte aligned
    template<typename Key, typename Type, typename Hash = std::hash<Key>, typename Pred = std::equal_to<Key> >
    using StdUMapAligned = std::unordered_map<Key, Type, Hash, Pred, Eigen::aligned_allocator<std::pair<const Key, Type> > >;

     // Special STL vectors where the type is 16byte aligned
    template<typename Type >
    using StdVecAligned = std::vector<Type, Eigen::aligned_allocator<Type> >;

    // Special Array types;
    using Array3 = Eigen::Array<PREC, 3, 1>;
    using Array2 = Eigen::Array<PREC, 2, 1>;
};


struct MyMatrixDecomposition {

    template<typename TMatrix>
    using EigenSolverSelfAdjoint = Eigen::SelfAdjointEigenSolver<TMatrix> ;

};

struct MyMatrixIOFormat {
    static Eigen::IOFormat Matlab;
    static Eigen::IOFormat CommaSep;
    static Eigen::IOFormat SpaceSep;
};


/**
* @brief This macro is used to typedef all custom matrix types which have nothing to do with the system.
*/
#define DEFINE_MATRIX_TYPES_OF( _PREC_ ) \
   using Matrix44 = typename MyMatrix< _PREC_ >::Matrix44; \
   using Matrix33 = typename MyMatrix< _PREC_ >::Matrix33; \
   using Matrix22 = typename MyMatrix< _PREC_ >::Matrix22; \
   using Matrix43 = typename MyMatrix< _PREC_ >::Matrix43; \
   using Matrix34 = typename MyMatrix< _PREC_ >::Matrix34; \
   using Vector3 = typename MyMatrix< _PREC_ >::Vector3;   \
   using Vector2 = typename MyMatrix< _PREC_ >::Vector2;   \
   using Vector4 = typename MyMatrix< _PREC_ >::Vector4;   \
   using Vector6 = typename MyMatrix< _PREC_ >::Vector6;   \
   using Quaternion = typename MyMatrix< _PREC_ >::Quaternion; \
   using AngleAxis = typename MyMatrix< _PREC_ >::AngleAxis; \
   using VectorDyn = typename MyMatrix< _PREC_ >::VectorDyn; \
   using MatrixDynDyn = typename MyMatrix< _PREC_ >::MatrixDynDyn; \
   using MatrixDiagDyn = typename MyMatrix< _PREC_ >::MatrixDiagDyn; \
   using MatrixDynDynRow = typename MyMatrix< _PREC_ >::MatrixDynDynRow; \
   \
   template<unsigned int M> using MatrixStatDyn = typename MyMatrix< _PREC_ >::template MatrixStatDyn<M>; \
   template<unsigned int N> using MatrixDynStat = typename MyMatrix< _PREC_ >::template MatrixDynStat<N>; \
   \
   template<typename Derived> using MatrixBase = typename MyMatrix< _PREC_ >::template MatrixBase<Derived>; \
   template<typename Derived> using ArrayBase  = typename MyMatrix< _PREC_ >::template ArrayBase<Derived>; \
   \
   template<typename Derived> using MatrixBDynDyn = typename MyMatrix< _PREC_ >::template MatrixBDynDyn<Derived>; \
   template<typename Derived, unsigned int N> using MatrixBDynStat = typename MyMatrix< _PREC_ >::template MatrixBDynStat<Derived,N>; \
   template<typename Derived, unsigned int M> using MatrixBStatDyn = typename MyMatrix< _PREC_ >::template MatrixBStatDyn<Derived,M>; \
   \
   using AffineTrafo = typename MyMatrix< _PREC_ >::AffineTrafo; \
   using AffineTrafo2d = typename MyMatrix< _PREC_ >::AffineTrafo2d; \
   using MatrixSparse = typename MyMatrix< _PREC_ >::MatrixSparse; \
   using MatrixSparseTriplet = typename MyMatrix< _PREC_ >::MatrixSparseTriplet; \
   \
   template<typename EigenType> using MatrixRef = typename MyMatrix< _PREC_ >::template MatrixRef< EigenType >; \
   template<typename EigenType> using MatrixMap = typename MyMatrix< _PREC_ >::template MatrixMap< EigenType >; \
   template<typename EigenType> using StdVecAligned = typename MyMatrix< _PREC_ >::template StdVecAligned< EigenType >; \
   template<typename Key, typename EigenType> \
   using StdMapAligned = typename MyMatrix< _PREC_ >::template StdMapAligned<Key, EigenType >; \
   template<typename Key, typename EigenType> \
   using StdUMapAligned = typename MyMatrix< _PREC_ >::template StdUMapAligned<Key, EigenType >; \
   \
   using Array3 = typename MyMatrix< _PREC_ >::Array3;   \
   using Array2 = typename MyMatrix< _PREC_ >::Array2;
#endif

