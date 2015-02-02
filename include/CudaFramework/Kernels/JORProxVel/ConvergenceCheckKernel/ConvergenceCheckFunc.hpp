#ifndef ConvergenceCheckFunc_hpp
#define ConvergenceCheckFunc_hpp
namespace ConvCheck{


template<typename PREC,typename BodyListType>
void calcConvCheck(BodyListType& bodyDataList){
    PREC value1;
    PREC value2;

    PREC relTol=0.1;
    PREC absTol=1;

    for(auto & d : bodyDataList) {

        value1 =  0.5*d.u.transpose()*d.regM*d.u;

        value2 =  0.5*d.u_2.transpose()*d.regM*d.u_2;

        d.test= (std::abs(value1-value2)<(relTol*value1+absTol));

    }
}
}

#endif



