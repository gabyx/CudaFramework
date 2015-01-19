==========
CudaFramework
==========

----------------------------------------
Projective Jacobi and Gauss-Seidel on the GPU for Non-Smooth Multi-Body Systems
----------------------------------------

This source code accompanies the paper   

> **G. Nützi et al. , Projective Jacobi and Gauss-Seidel on the GPU for Non-Smooth Multi-Body Systems, 2014** , download : [1](http://proceedings.asmedigitalcollection.asme.org/proceeding.aspx?articleID=2091012) or [2](http://www.zfm.ethz.ch/~nuetzig/_private_files/projective.pdf)

The [master thesis](http://dx.doi.org/10.3929/ethz-a-010054012) should be consulted only in the case of being interested in the details of certain GPU variants (see below).

---------------------------
Installation & Dependencies
---------------------------
To build the performance tests (MatrixMultiply, Prox, etc.) you need the built tool [cmake](
http://www.cmake.org).
The performance tests only depend on the matrix library [Eigen](http://eigen.tuxfamily.org) at least version 3. Download it and install it on your system.
You need CUDA installed on your system as well, download and install the latest [CUDA API and Drivers](https://developer.nvidia.com/cuda-downloads).

Download the latest CudaFramework code:
```bash
    $ git clone https://github.com/gabyx/CudaFramework.git CudaFramework  
```
Make a build directory and navigate to it:
```bash
    $ mkdir Build
    $ cd Build
```
Invoke cmake in the Build directory:
```bash
    $ cmake ../CudaFramework
```
The cmake script will find Eigen and Cuda if you installed it in a system wide folder (e.g ``/usr/local/``)


Finally, build the performance tests:
```bash
    $ make all 
    $ make install
``` 
 To build in parallel use the ``-jN`` flag in the `make` commmand, where ``N``denotes the number of parallel threads to use.

--------------------------
Supported Platforms
--------------------------
The code has been tested on Linux and OS X with compilers ``clang`` and ``gcc``. 
It should work for Windows as well, but has not been tested.

---------------------------
Example Usage
---------------------------
The performance tests are all written in the same structure. 
A performance tests of any kind of application can be specified with the ``PerformanceTest`` template class which accepts a test method as template argument.
For the kernel performance tests mainly used in this project, the test method ``KernelTestMethod`` is of main interest.   

The target ``PerformanceProx`` contains the parallel GPU implementation of the projective overrelaxed Jacobi (JORProx) and succesive overrelaxed Gauss-Seidel (SORProx, SORProxRelaxed) iterations used in multi-body dynamics.

The target ``PerformanceMatrix`` contains the performance test of the efficient parallel matrix-multiplication kernel which is used for the JORProx implementation.

The following example shows how a performance test for the SORProx GPU Variant 1 is launched (target: ``PerformaceProx``):
```C++
    typedef KernelTestMethod< /* Specify a kernel test method */ 
    
        KernelTestMethodSettings<false,true,0> ,  /* Specify the general kernel test method settings */ 
        
        ProxTestVariant< /* Specify the specific variant of the kernel test method */ 
            
            ProxSettings< /* All settings for the prox test variant */ 
        
               double,                  /* use double floating point precision */ 
               SeedGenerator::Fixed<5>, /* the seed for the random value generator for the test data */ 
               false,                   /* Write the test data to a file (matlab style) */ 
               1,                       /* Max. iterations of the prox iteration */
               false,                   /* Abort iteration if prox iteration converged */
               10,                      /* Convergence check every 10 iterations */
               true,                    /* Compare the GPU implementation to the exact serial replica on th CPU */
               ProxPerformanceTestSettings<3000,20,4000,3>,                /* Problem sizes from 3000 contacts to 4000 in steps of 20, generate 3 random test problems per problem size*/
               SorProxGPUVariantSettings<1,ConvexSets::RPlusAndDisk,true>  /* Use the GPU Variant 1, align the memory on the GPU for coalesced access!*/
            >
            
        >
        
    > test1;
    
    PerformanceTest<test1> A("SorProxVariant1D");
    A.run();
```

To understand the JORProx and SORProx, the user is encouraged to understand the workflow of the ``ProxTestVariant`` in ``ProxTestVariant.hpp``.
This class contains the basic initialization of the used matrices for the numerical iterations, the two important functions
``ProxTestVariant::runOnGPU()`` and ``ProxTestVariant::runOnCPU()`` which run the specified variant (e.g. SORProx GPU Variant 1 in the above example) 
on the CPU or the GPU, and a check routine
``ProxTestVariant::checkResults()`` which compares the results from the GPU to the CPU.
The function call ``ProxTestVariant::runOnGPU()`` calls the ``runGPUProfile()`` function of the templated type ``ProxTestVariant::m_gpuVariant``.

The GPU variants for the type ``m_gpuVariant`` of the JORProx and SORProx can be found in ``SorProxGPUVariant.hpp`` and ``JorProxGPUVariant.hpp``. 
These files will help the most in understanding the source code together with the paper.

Each GPU variant class ``JorProxGPUVariant`` and ``SorProxGPUVariant`` contains certain variants which correspond to fixed GPU settings (block dimension, threads per block etc...).
The descriptions of these variants are consistent with the master thesis (and hopefully also the paper).
Each GPU variant has a ``initializeTestProblem()`` function which fills the iteration matrices with random values (keeping the problem size fixed!).
Each GPU variant also has ``runGPUProfile()`` and ``runGPUPlain()`` functions which launch the GPU variants with or without timing information.

To get to the bottom of the prox iteration variants, consider the the kernels A and B involved in the GPU variant ``SorProxGPUVariant``. These are launched sequentially over the iteration matrix ``T_dev`` as shown in the following:
```C
    for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++){

               //cudaThreadSynchronize();
               proxGPU::sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings1>(
                  mu_dev,x_new_dev,T_dev,d_dev,
                  t_dev,
                  kernelAIdx,
                  pConvergedFlag_dev,
                  m_absTOL,m_relTOL);

               proxGPU::sorProx_StepB_kernelWrap<SorProxSettings1>(
                  t_dev,
                  T_dev,
                  x_new_dev,
                  kernelAIdx
                  );

            }
```

**Interfacing with Own Code:**
The best way to use the SORProx or JORProx GPU implementations right out of the box is to instantiate the following
variant types somewhere in your code:
```C++
   JorProxGPUVariant< JorProxGPUVariantSettingsWrapper<PREC,5,ConvexSets::RPlusAndDisk,true,300,true,10,false, TemplateHelper::Default>, ConvexSets::RPlusAndDisk > m_jorGPUVariant;

   SorProxGPUVariant< SorProxGPUVariantSettingsWrapper<PREC,1,ConvexSets::RPlusAndDisk,true,300,true,10,true,  TemplateHelper::Default >,  ConvexSets::RPlusAndDisk > m_sorGPUVariant;
```

and then launching the iterations with something like this:

```C++
        m_jorGPUVariant.setSettings(m_settings.m_MaxIter,m_settings.m_AbsTol,m_settings.m_RelTol);
        gpuSuccess = m_jorGPUVariant.runGPUPlain(P_front,m_T,P_back,m_d,m_mu);
        m_globalIterationCounter = m_jorGPUVariant.m_nIterGPU;
        /* OR */
        m_sorGPUVariant.setSettings(m_settings.m_MaxIter,m_settings.m_AbsTol,m_settings.m_RelTol);
        gpuSuccess = m_sorGPUVariant.runGPUPlain(P_front,m_T,P_back,m_d,m_mu);
        m_globalIterationCounter = m_jorGPUVariant.m_nIterGPU;
```
Matrices ``m_T`` and ``m_d`` are built as described in the paper. Vector ``m_mu`` are the friction coefficients for all contacts which consist of a normal and two tangential forces. The percussions ``P_back`` and ``P_front`` are contact ordered and each contact tuple consits of (normal percussion, tangential percussion 1, tangential percussion 2).


--------------------------
Licensing
--------------------------

This source code is released under GNU GPL 3.0. 

---------------------------
Author and Acknowledgements
---------------------------

CudaFramework was written by Gabriel Nützi. Source code from [ModernGPU](http://www.moderngpu.com) has been used, see ``CudaModern`` folder in ``/include/CudaFramework``
