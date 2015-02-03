
#ifndef CudaFramework_Kernels_JORProxVel_GenRandomContactGraphClass_hpp
#define CudaFramework_Kernels_JORProxVel_GenRandomContactGraphClass_hpp

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"

template<typename VectorIntType,typename MatrixType,typename MatrixUIntType>
class GenRndContactGraph {
public:
     struct ContactType {

        unsigned int bodyIdx1;
        unsigned int bodyIdx2;

        unsigned int redIndexBdy1;
        unsigned int redIndexBdy2;

    };

    template <typename Stack>
    static void CheckCorrectness(Stack contactList,unsigned int numberOfBodies) {
        std::vector<unsigned int> buffer;
        buffer.resize(numberOfBodies);

        for(auto it = contactList.begin(); it != contactList.end(); ++it) {

            buffer[it->bodyIdx1]++;
            buffer[it->bodyIdx2]++;

        }

        for(auto it = buffer.begin(); it != buffer.end(); ++it) {

            if((*it)==0) {
                std::cout<< "ERROR: Irgendein Körper hat keinen Kontakt"<<std::endl;
            }
        }
    }

    template<typename Stack,typename ContactVec>
    static void makecontact(unsigned int bodyIdx1,
                     unsigned int bodyIdx2,
                     Stack &contactList,
                     unsigned int contactNumber,
                     ContactVec& contacts) {

        ContactType newContact;
        newContact.bodyIdx1=bodyIdx1;
        newContact.bodyIdx2=bodyIdx2;
        contacts.push_back(newContact);

        contactList[bodyIdx1].push_back(contactNumber);
        contactList[bodyIdx2].push_back(contactNumber);

    }


public:

     /**  Pyramid scheme **/

    struct pyramidScheme {
        template<typename Stack,typename ContactListe>
        static void eval(unsigned int numberOfBodies,
        unsigned int counter,
        Stack& contactList,
        ContactListe& contacts) {


            unsigned int contactNumber=0;

            for(int i=0; i<numberOfBodies-1; i++) {

                std::default_random_engine generator;
                std::uniform_int_distribution<int> distribution(i+1,numberOfBodies);

                for(int j=0; j<counter; j++) {

                      makecontact(i,distribution(generator),contactList,contactNumber,contacts);
                     contactNumber++;
                }
            }
        }

        static void getContNum(unsigned int numberOfBodies,
        unsigned int counter,
        unsigned int & contactNum){
        contactNum=counter*(numberOfBodies-1);

      }



    };





    struct bodySerpentScheme {
        template<typename Stack,typename ContactListe>
        static void eval(unsigned int numberOfBodies,
        unsigned int counter,
        Stack& contactList,
        ContactListe& contacts) {


            unsigned int contactNumber=0;

            for(int i=0; i<numberOfBodies-1; i++) {

                for(int j=0; j<counter; j++) {

                      makecontact(i,i+1,contactList,contactNumber,contacts);
                     contactNumber++;
                }
            }
        }

        static void getContNum(unsigned int numberOfBodies,
        unsigned int counter,
        unsigned int & contactNum){
        contactNum=counter*(numberOfBodies-1);

      }



    };
      /**    Contact Permutation **/

    struct randomSeg2 {
        template<typename Stack,typename ContactListe>
        static void eval(unsigned int numberOfBodies,
        unsigned int counter,
        Stack& contactList,
        ContactListe& contacts) {

            std::vector<unsigned int> vec_1(numberOfBodies,0);
            for(int i=0; i<numberOfBodies; i++) {
                vec_1[i]=i;
            }

            std::random_shuffle(vec_1.begin(),vec_1.end());
            for(int j=0; j<numberOfBodies; j++) {

                if((vec_1[j]==j )) {
                    if(j==0) {
                        std::swap(vec_1[j],vec_1[j+1]);
                    } else {
                        std::swap(vec_1[j],vec_1[j-1]);
                    }
                }

            }
            unsigned int contactNumber=0;
            for(int i=0; i<counter; i++) {
                for(int j=0; j<numberOfBodies; j++) {

                    makecontact(j,vec_1[j],contactList,contactNumber,contacts);
                    contactNumber++;
                }
            }
        }
        static void getContNum(unsigned int numberOfBodies,
                               unsigned int counter,
                               unsigned int & contactNum) {
            contactNum=counter*numberOfBodies;

        }

    };


  /**    Each body gets ''counter'' random contacts **/


    struct randomSeg3 {
        template<typename Stack,typename ContactListe>
        static void eval(unsigned int numberOfBodies,
        unsigned int counter,
        Stack& contactList,
        ContactListe& contacts) {


            std::default_random_engine generator;
            std::uniform_int_distribution<unsigned int> distribution(0,numberOfBodies);

            unsigned int bodyIdx2;
            unsigned int contactNumber=0;
            std::vector<unsigned int> vec_1(numberOfBodies,0);
            for(int j=0; j<counter; j++) {
                for(int i=0; i<numberOfBodies; i++) {
                    bodyIdx2=distribution(generator);
                    while(bodyIdx2==i) {
                        bodyIdx2=distribution(generator);
                    }
                    makecontact(j,vec_1[j],contactList,contactNumber,contacts);
                    contactNumber++;
                }
            }
        }


        static void getContNum(unsigned int numberOfBodies,
        unsigned int counter,
        unsigned int & contactNum){
        contactNum=counter*numberOfBodies;

      }
    };







    template<typename Stack,typename VectorCsr>
    static int generateCSR(Stack contactList,VectorCsr &outCSR) {

        outCSR[0]=0;
        for(unsigned int i=0; i<contactList.size()-1; i++) {
            outCSR[i+1]=contactList[i].size()+outCSR[i];
        }
        return 0;

    }



    template<typename ContactVec,typename Stack >
    static void fillContactStruct(ContactVec &contacts,
                           Stack contactList) {
        int number;
        int counter=0;
        for(unsigned int i=0; i<contactList.size(); i++) {
            for(unsigned int j=0; j<contactList[i].size(); j++) {
                number=contactList[i][j];
                if(contacts[number].bodyIdx1==i) {
                    contacts[number].redIndexBdy1=counter;
                } else {
                    contacts[number].redIndexBdy2=counter;
                }
                counter++;
            }

        }
    }




    template<typename Func,bool genLog=true>
    static void generateRandomSetup (unsigned int numberOfBodies,
                              unsigned int minContactsPerBody,
                              VectorIntType & csrArray,
                              MatrixType & contactMatrix,
                              MatrixUIntType & redIndexMatrix ) {

        if(numberOfBodies<2) {
            std::cout<< "Fehler weniger als 2 Körper, kontaktgraph wurde fuer 2 Körper erstellt !! " <<std::endl;
            numberOfBodies=2;
        }

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        std::vector<std::vector<unsigned int> > contactList;
        std::vector<ContactType> contacts;


        unsigned int numberOfContacts=minContactsPerBody*numberOfBodies;



        contactList.resize(numberOfBodies);
        csrArray.resize(numberOfBodies);

        Func::eval(numberOfBodies,
                        minContactsPerBody,
                        contactList,
                        contacts);

        if(genLog) {
            std::cout<<"size 1: "<<contactList.size()<<std::endl;
            std::cout<<"size 2: "<<contactList[0].size()<<std::endl;

            for(int unsigned i=0; i<contactList.size(); i++) {
                std::cout<<"Body nummer: "<<i<<std::endl;
                for(unsigned int j=0; j<contactList[i].size(); j++) {
                    std::cout<<"Kontakt nummer: "<<j<<std::endl;
                    std::cout<<"Hat die nummer: "<<contactList[i][j]<<std::endl;
                }
            }
            std::cout<<"Element aus der contactList"<<   contactList[0][0]<<std::endl;
        }

        generateCSR(contactList,csrArray)/2;

        contacts.resize(numberOfContacts);

        fillContactStruct(contacts,contactList);

        CheckCorrectness(contacts,numberOfBodies);


        if(genLog) {
            for(unsigned int i=0; i<contacts.size(); i++) {
                std::cout<<"Kontakt nummer: "<<i<<std::endl;
                std::cout<<"bodyIdx1: "<<contacts[i].bodyIdx1<<std::endl;
                std::cout<<"bodyIdx2: "<<contacts[i].bodyIdx2<<std::endl;
                std::cout<<"Reduktions Index 1: "<<contacts[i].redIndexBdy1<<std::endl;
                std::cout<<"Reduktions Index 2: "<<contacts[i].redIndexBdy2<<std::endl;
            }
        }
        contactMatrix.resize(contacts.size(),C::length);
        redIndexMatrix.resize(contacts.size(),I::length);

        for(unsigned int i=0; i<contacts.size(); i++) {
            redIndexMatrix(i,I::redIdx_s)=contacts[i].redIndexBdy1;
            redIndexMatrix(i,I::redIdx_s+1)=contacts[i].redIndexBdy2;
            redIndexMatrix(i,I::b1Idx_s)=contacts[i].bodyIdx1;
            redIndexMatrix(i,I::b2Idx_s)=contacts[i].bodyIdx2;
        }
    }

};


#endif // GenRandomContactsGraph
