// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_PerformanceTest_KernelTestMethod_hpp
#define CudaFramework_PerformanceTest_KernelTestMethod_hpp


#include <iostream>
#include <fstream>
#include <sstream>

#include <pugixml.hpp>

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaUtilities.hpp"



#define CHECK_XMLNODE( _node_ , _nodename_ ) \
    if( ! _node_ ){ \
        THROWEXCEPTION("XML Node: " << _nodename_ << " does not exist!");  \
    }
#define CHECK_XMLATTRIBUTE( _node_ , _nodename_ ) \
    if( ! _node_ ){ \
        THROWEXCEPTION("XML Attribute: " << _nodename_ << " does not exist!");  \
    }

#define GET_XMLCHILDNODE_CHECK( _childnode_ , _childname_ , _node_ ) \
    _childnode_ = _node_.child( _childname_ ); \
    CHECK_XMLNODE( _childnode_ , _childname_)

#define GET_XMLATTRIBUTE_CHECK( _att_ , _attname_ , _node_ ) \
    _att_ = _node_.attribute( _attname_ ); \
    CHECK_XMLATTRIBUTE( _att_ , _attname_ )

/**
* @addtogroup PerformanceTest
* @{
*/


/**
* @defgroup PerformanceTestMethods Performance Test Methods
* @brief These are the Performance Test Methods which can be launched with the PerformanceTest class, by inserting this template class as its argument.
* @{
*/


/**
* @defgroup KernelTestMethod Kernel Test Method
* @brief This is an implementation to test a certain TTestVariant.
* The test runs different Test Problems, and each Test Problem runs a certain number of random runs.
* This Method collects timings of CPU an GPU implementation of the TTestVariant, (GFlops, Bandwith,Copy Time GPU, Speed Up, Tradoffs and more...), it does also a Result Check if specified.
*
* It writes all result with the used GPU specification to a Data file! The Log goes either to the console or to a file.
*
* How to launch this Test Method:
* \code

   #include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
   #include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultTestVariant.hpp"

   typedef KernelTestMethod<                                         // Use the KernelTestMethod!
         KernelTestMethodSettings<false,true> ,                      // Some settings for the KernelTestMethod!
         MatrixVectorMultTestVariant<                                // The Test Variant which should be tested with the KernelTestMethod
            MatrixVectorMultSettings<                                   // Some settings for the Test Variant
               double,                                                  // Use double
               3,                                                       // ContactSize = 3
               false,                                                   // Dont write to file!
               10,                                                      // max iterations for the multiplication
               MatrixVectorMultPerformanceTestSettings<0,20,2000,5>,    // Some Performance settings, how to iterate over all test problems, and over the random runs!
               MatrixVectorMultGPUVariantSettings<2,true>               // The actual GPUVariant 2 for this Test Variant, alignMatrix is true
            >
         >
    > test1;

   PerformanceTest<test1> A("test1");  // Define a Performance Test!
   A.run();                            // Run the Test!
* \endcode
* @{
*/

/*
* @brief These are the settings for this Test Method.
*/
template<bool logToFile, bool checkResults, int gpuDeviceToUse = 0>
struct KernelTestMethodSettings {
    static const bool LogToFile = logToFile;
    static const bool CheckResults =  checkResults;
    static const int UseGPUDeviceID = gpuDeviceToUse;
};

#define DEFINE_KernelTestMethod_Settings(_TSettings_) \
   static const bool LogToFile = _TSettings_::LogToFile; \
   static const bool CheckResults =  _TSettings_::CheckResults; \
   static const int UseGPUDeviceID = _TSettings_::UseGPUDeviceID;


template<typename TKernelTestMethodSettings, typename TTestVariant>
class KernelTestMethod {
public:

    DEFINE_KernelTestMethod_Settings(TKernelTestMethodSettings);
    typedef typename TTestVariant::PREC PREC;


    using XMLDocumentType = pugi::xml_document;
    using XMLNodeType = pugi::xml_node;
    static const auto XMLStringNode = pugi::node_pcdata;

    KernelTestMethod():
        m_oData(std::cout.rdbuf()),
        m_oLog(std::cout.rdbuf()) {
    }

    ~KernelTestMethod() {}

    void initialize(std::string test_name) {

        m_test_name = test_name;

        if(LogToFile) {
            m_ofsLog.open("Log.txt");
            m_ofsData.close();
            if (m_ofsLog.is_open()) {
                std::cout << " File opened: " << "Log.txt" <<std::endl;
            }
            m_oLog.rdbuf(m_ofsLog.rdbuf());
        } else {
            m_oLog.rdbuf(std::cout.rdbuf());
        }

        m_filename = m_test_name + TTestVariant::getTestVariantDescription();

        m_ofsData.close();
        m_ofsData.open(m_filename+".dump");
        if (m_ofsData.is_open()) {
            std::cout << " File opened: " << m_filename <<std::endl;
        } else {
            ERRORMSG("Could not open data file: " << m_filename);
        }
        m_oData.rdbuf(m_ofsData.rdbuf());


        // Set GPU Device to use!
        m_oLog << " Set GPU Device: " << UseGPUDeviceID <<std::endl;
        cudaDeviceReset();
        CHECK_CUDA(cudaSetDevice(UseGPUDeviceID));


        m_testVariant.initialize(&m_oLog,&m_oData);

        m_oLog << " Kernel Performance Test  ======================================="<<std::endl;
        m_oData<< "# Kernel Performance Test:  Data Dump for file:" <<m_filename<<".xml" << std::endl;

        // Init XML
        m_dataXML.reset();
        std::stringstream xml("<PerformanceTest type=\"KernelTest\">"
                              "<Description></Description>"
                              "<DataTable>"
                              "<Header></Header>"
                              "<Data></Data>"
                              "</DataTable>"
                              "</PerformanceTest>");

        bool r = m_dataXML.load(xml);
        ASSERTMSG(r,"Could not load initial xml data file");

        // DESCRIPTION ==========================
        // Add GPU INFO
        XMLNodeType descNode = m_dataXML.child("PerformanceTest").child("Description");
        // Write header to file for this TestMethod!
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props,UseGPUDeviceID));
        std::stringstream s;
        utilCuda::writeCudaDeviceProbs(s,props,UseGPUDeviceID);
        auto gpuInfo = descNode.append_child("GPUInfos").append_child(XMLStringNode);
        gpuInfo.set_value(s.str().c_str());


        // Write variant specific descriptions!
        auto desc = m_testVariant.getDescriptions();
        for(auto & d : desc){
            auto gpuInfo = descNode.append_child(d.first.c_str()).append_child(XMLStringNode);
            gpuInfo.set_value(d.second.c_str());
        }

        // DATA =========================================
        XMLNodeType dataTable = m_dataXML.child("PerformanceTest").child("DataTable");
        auto headerNode = dataTable.child("Header");
        // Write variant specific header
        {
            std::vector<std::string> head = m_testVariant.getDataColumHeader();
            for(auto & h : head){
                auto col = headerNode.append_child("Column").append_child(XMLStringNode);
                col.set_value(h.c_str());
            }
        }
        {
            // Write TestMethod Specific Column Header!
            std::vector<std::string> head;
            head.push_back("nFlop");
            head.push_back("GFlops");
            head.push_back("Memory Bandwith [Bytes/sec]");
            head.push_back("elapsedTimeCopyToGPU_Avg [s]");
            head.push_back("gpuIterationTime_Avg [s]");
            head.push_back("elapsedTimeCopyFromGPU_Avg [s]");
            head.push_back("cpuIterationTime_Avg [s]");
            head.push_back("nIterationsForTradeoff");
            head.push_back("speedUpFactor");
            head.push_back("maxRelTol_Avg (over all TestProblems)");
            head.push_back("avgRelTol_Avg (over all TestProblems)");
            head.push_back("maxUlp_Avg (over all TestProblems)");
            head.push_back("avgUlp_Avg (over all TestProblems)");
            for(auto & h : head){
                auto col = headerNode.append_child("Column").append_child(XMLStringNode);
                col.set_value(h.c_str());
            }
        }

        // Save XML already for savety!
        m_dataXML.save_file((m_filename+".xml").c_str(),"    ");


    }

    void runTest() {

        using namespace boost;

        while(m_testVariant.generateNextTestProblem()) {

            double cpuIterationTime_Avg         = 0;
            double gpuIterationTime_Avg         = 0;
            double elapsedTimeCopyToGPU_Avg     = 0;
            double elapsedTimeCopyFromGPU_Avg   = 0;
            typename TypeWithSize<sizeof(PREC)>::UInt maxUlp_Avg = 0;
            double avgUlp_Avg    = 0;
            double maxRelTol_Avg = 0;
            double avgRelTol_Avg = 0;



            while(m_testVariant.generateNextRandomRun()) {


                m_testVariant.runOnGPU();
                elapsedTimeCopyToGPU_Avg += m_testVariant.m_elapsedTimeCopyToGPU;
                gpuIterationTime_Avg += m_testVariant.m_gpuIterationTime;
                elapsedTimeCopyFromGPU_Avg += m_testVariant.m_elapsedTimeCopyFromGPU;


                m_testVariant.runOnCPU();
                cpuIterationTime_Avg += m_testVariant.m_cpuIterationTime;

                if(CheckResults) {
                    m_testVariant.checkResults();
                    maxRelTol_Avg += m_testVariant.m_maxRelTol;
                    avgRelTol_Avg += m_testVariant.m_avgRelTol;
                    maxUlp_Avg += m_testVariant.m_maxUlp;
                    avgUlp_Avg += m_testVariant.m_avgUlp;
                }

            }

            m_testVariant.cleanUpTestProblem();

            // Average values
            cpuIterationTime_Avg /= TTestVariant::maxNRandomRuns;
            gpuIterationTime_Avg /= TTestVariant::maxNRandomRuns;
            elapsedTimeCopyFromGPU_Avg /= TTestVariant::maxNRandomRuns;
            elapsedTimeCopyToGPU_Avg /= TTestVariant::maxNRandomRuns;
            maxRelTol_Avg /= TTestVariant::maxNRandomRuns;
            avgRelTol_Avg /= TTestVariant::maxNRandomRuns;
            maxUlp_Avg /= TTestVariant::maxNRandomRuns;
            avgUlp_Avg /= TTestVariant::maxNRandomRuns;

            double speedUpFactor  = cpuIterationTime_Avg / gpuIterationTime_Avg;
            m_oLog << "Speed Up Factor 1 Iteration GPU: "<< speedUpFactor  << "x faster"<<std::endl;

            double preprocessTimeGPU_Avg = elapsedTimeCopyToGPU_Avg + elapsedTimeCopyFromGPU_Avg /*+ other stuff*/ ; // ms
            double nIterationsForTradeoff = (preprocessTimeGPU_Avg)/(cpuIterationTime_Avg  - gpuIterationTime_Avg);
            double nOps = m_testVariant.m_nOps;
            double GFlops = (nOps / (gpuIterationTime_Avg*1.0e-3)) * 1.0e-9;
            double Bandwith = m_testVariant.m_nBytesReadWrite / (gpuIterationTime_Avg*1.0e-3); // Bytes /sec

            if(nIterationsForTradeoff < 0) {
                nIterationsForTradeoff = 0;
            }
            m_oLog << "GPU is prevalent for size if more than " << (int)nIterationsForTradeoff << " iterations are performed on GPU" <<std::endl;




            //Write Variant specific columns of data into the file
            m_testVariant.writeData();

            // Write TestMethod specific columns to file
            m_oData << boost::format("%1$.9d\t%2$.9d\t%3$.9d\t%4$.9d\t%5$.9d\t%6$.9d\t%7$.9d\t%8$.9d\t%9$.9d")
                    % nOps
                    % GFlops
                    % Bandwith
                    % (elapsedTimeCopyToGPU_Avg*1e-3)
                    % (gpuIterationTime_Avg*1e-3)
                    % (elapsedTimeCopyFromGPU_Avg*1e-3)
                    % (cpuIterationTime_Avg*1e-3)
                    % nIterationsForTradeoff
                    % speedUpFactor;

            if(CheckResults) {
                m_oData << boost::format("\t%1$.9d\t%2$.9d\t%3$.9d\t%4$.9d")
                        % avgRelTol_Avg
                        % maxRelTol_Avg
                        % avgUlp_Avg
                        % (double)maxUlp_Avg <<std::endl;
            } else {
                m_oData << boost::format("\t%1$.9d\t%2$.9d\t%3$.9d\t%4$.9d")
                        % -0.0
                        % -0.0
                        % -0.0
                        % -0.0  <<std::endl;
                m_oData<< std::endl;
            }

        }



    }

    void finalize() {

        m_testVariant.finalize();

        m_ofsLog.close();
        m_ofsData.close();

        // Read in DataTable again and save it in the XML!

        std::ifstream t(m_filename+".dump");
        std::stringstream dataTable;
        dataTable << t.rdbuf();

        // Write back to datatable
        auto dataNode = m_dataXML.child("PerformanceTest").child("DataTable").child("Data").append_child(XMLStringNode);
        dataNode.set_value(dataTable.str().c_str());
        m_dataXML.save_file((m_filename+".xml").c_str(),"    ");

        // Delete dump file:
        std::remove((m_filename+".dump").c_str());
    }

private:
    std::string m_test_name;

    std::string m_filename;
    std::ofstream m_ofsLog;
    std::ofstream m_ofsData;
    std::ostream m_oData;
    std::ostream m_oLog;

    XMLDocumentType m_dataXML;


    TTestVariant m_testVariant;
};


/** @defgroup TestVariants Test Variants */


/** @} */ //Kernel Test Method


/** @} */ //Performance Test Methods



/** @} */

#endif

