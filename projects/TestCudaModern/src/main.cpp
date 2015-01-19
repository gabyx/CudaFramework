
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "CudaFramework/CudaModern/CudaContext.hpp"
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"
#include "CudaFramework/CudaModern/CudaPrint.hpp"

int main(){


    //general function

    //utilCuda::printAllCudaDeviceSpecs();


{
    utilCuda::ContextPtrType ptr = utilCuda::createCudaContextOnDevice(0);
    ptr->setActive();

    std::cout << ptr->device().deviceString() << std::endl;

    //Make a managed device memory pointer (intrusive pointer, cleans it self)
    std::vector<utilCuda::DeviceMemPtr<char> > pDevs;
    for(int i=0;i<10;i++){
        pDevs.push_back( ptr->malloc<char>(10<<20) );
        //std::cout << static_cast<utilCuda::CudaAllocBuckets*>(ptr->getAllocator())->allocated() << std::endl;
    }

    pDevs.erase(pDevs.begin(),pDevs.begin()+2);


    std::cout << ptr->device().memInfoString() << std::endl;
    pDevs.push_back( ptr->malloc<char>(2000<<20) ); // To big for a allocator bucket, goes into uncached!
    pDevs.pop_back();
    std::cout << ptr->device().memInfoString() << std::endl;

    // Make another
    utilCuda::DeviceMemPtr<int> pDev2 = ptr->genRandom<int>(100,4,10);
    utilCuda::printArray(*pDev2,"%4d", 10);

    pDevs.clear();
    std::cout << ptr->device().memInfoString() << std::endl;
}


std::cout << utilCuda::CudaDevice::selected().memInfoString() << std::endl;
// Memory Stress Test
{
    utilCuda::ContextPtrType ptr = utilCuda::createCudaContextOnDevice(0);
    ptr->setActive();
    std::vector<utilCuda::DeviceMemPtr<char> > pDevs;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,300);
    int dice_roll = distribution(generator);

    for(int l=0;l<200;l++){
        //add
        for(int i=0;i<5;i++){
                std::cout << "add" <<",";
            pDevs.push_back( ptr->malloc<char>(distribution(generator)<<20) );
        }
        std::cout << static_cast<utilCuda::CudaAllocBuckets*>(ptr->getAllocator())->capacity();
        //remove
        for(int i=0;i<5;i++){
            std::cout << "remove" << ",";
            pDevs.pop_back();
        }
        std::cout << static_cast<utilCuda::CudaAllocBuckets*>(ptr->getAllocator())->capacity();
    }
    std::cout << std::endl;
}
std::cout << utilCuda::CudaDevice::selected().memInfoString() << std::endl;


{
    utilCuda::ContextPtrType c = utilCuda::createCudaContextOnDevice(0,false);
    c->setActive();

     // Make another
     {
         utilCuda::DeviceMatrixPtr<double> pA_dev = c->mallocMatrix<double,false>(13000,13000);
         std::cout << "Size of CudaMatrix: " << sizeof(*pA_dev) << std::endl;
         std::cout << "pitch:" << pA_dev->get().m_outerStrideBytes << std::endl;
         std::cout << c->device().memInfoString() << std::endl;
     }

     // Make another aligned matrix :)
     {
         std::cout << c->device().memInfoString() << std::endl;
         utilCuda::DeviceMatrixPtr<double> pA_dev = c->mallocMatrix<double,true>(13000,2);
         std::cout << "pitch:" << pA_dev->get().m_outerStrideBytes << std::endl;
         std::cout << c->device().memInfoString() << std::endl;

         Eigen::MatrixXd A(13000,2);

         std::cout << "Copy From Matrix" <<std::endl;
         CHECK_CUDA(pA_dev->fromHost(A));

         std::cout << "Copy From temporary expression A+A" <<std::endl;
         CHECK_CUDA(pA_dev->fromHost(A+A));

         Eigen::MatrixXd B(13000,4);
//
         std::cout << "Copy From temporary expression (block)" <<std::endl;
         CHECK_CUDA(pA_dev->fromHost(B.leftCols(2)));


         Eigen::MatrixXd C(13400,4);

         std::cout << "Copy From temporary expression (block)" <<std::endl;
         CHECK_CUDA(pA_dev->fromHost(C.block(400,2,13000,2)));

         std::cout << "Copy From temporary expression (block)" <<std::endl;
         //CHECK_CUDA(pA_dev->fromHost(c->block(30,2,12500,2))); // fails because not rigth size!

     }


     {
         utilCuda::DeviceMatrixPtr<double> pA_dev = c->mallocMatrix<double,true>(55,5);
         Eigen::MatrixXd A(55,5);
         A.setOnes();
         std::cout << "A: " << std::endl << A << std::endl;
         pA_dev->fromHost(A+A);
         utilCuda::printArray(*pA_dev,"%4f"); // copies internally the matrix to the device before outputing!
     }


}
    // see how the system changes the GPU mem! :-)
    for(int i=0; i < 10; i++){
        sleep(1);
        std::cout << utilCuda::CudaDevice::selected().memInfoString() << std::endl;
    }

    utilCuda::destroyDeviceGroup();

};
