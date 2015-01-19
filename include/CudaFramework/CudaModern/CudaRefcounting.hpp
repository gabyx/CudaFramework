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

#ifndef CudaFramework_CudaModern_CudaRefcounting_hpp
#define CudaFramework_CudaModern_CudaRefcounting_hpp

#include <utility>

#include <type_traits>

namespace utilCuda{


    class NonCopyable {
    protected:
        NonCopyable() {}
        ~NonCopyable() {}
    private:
        NonCopyable(const NonCopyable&) { }
        const NonCopyable& operator=(const NonCopyable&) { return *this; }
    };

    //Prototypes
    template<typename TPreDerived, typename TDerived>
	class ReferenceCounting;
    template<typename T> class IntrusivePtr;
    template<typename Derived,typename PreDerived>
    void addRefIntrusivPtr( ReferenceCounting<PreDerived,Derived> * p);
    template<typename Derived,bool Delete, typename PreDerived >
    void releaseIntrusivePtr( ReferenceCounting<PreDerived,Derived> * p);

    // TPreDerived is the derived of TDerived, and TDerived is derived of this ReferenceCounting Class
	template<typename TPreDerived, typename TDerived>
	class ReferenceCounting : public NonCopyable{
    private:
    // If TPreDerived is void , we need to delete TDerived, otherwise the TPreDerived
        using DerivedToDelete = typename std::conditional< std::is_same<TPreDerived,void>::value,
                                                  TDerived, TPreDerived
                                                >::type;
    public:

        ReferenceCounting() : m_ref(0) { }
        // Do not change reference count if an assignment has been done
        ReferenceCounting& operator= (ReferenceCounting const&){ return *this; }

        unsigned long int getRefCount() const{return m_ref;}

    protected:
    	~ReferenceCounting() {/*std::cout <<"RC:DTOR" <<std::endl;*/ }

    private:

    	friend  class IntrusivePtr<TDerived>;
    	friend  class IntrusivePtr<const TDerived>;

    	template<typename Derived,typename PreDerived>
        friend void addRefIntrusivPtr( ReferenceCounting<PreDerived,Derived> * p);
        template<typename Derived,bool Delete, typename PreDerived >
        friend void releaseIntrusivePtr( ReferenceCounting<PreDerived,Derived> * p);

    	unsigned long int addRef() const{
    		++m_ref;
    		//std::cout << "RC::addRef: " << m_ref <<  std::endl;
    		return m_ref;
    	}

    	// NoDelete is for IntrusivePtr<T>().release()!
    	template<bool Delete = true>
    	void release() const{
    		--m_ref;
    		//std::cout << "RC::release: " <<  m_ref << std::endl;
    		if(!m_ref && Delete){
    			//std::cout << "RC::delete" << std::endl;
    			delete static_cast<DerivedToDelete const *>(this);
    		}
    	}

        mutable unsigned long int m_ref; // Mutable to be changeable also for const objects!
    };


    template<typename Derived,typename PreDerived>
    inline void addRefIntrusivPtr( ReferenceCounting<PreDerived,Derived> * p){
        p->addRef();
    }

    template<typename Derived,bool Delete = true, typename PreDerived >
    inline void releaseIntrusivePtr( ReferenceCounting<PreDerived,Derived> * p){
        p->template release<Delete>();
    }

    template<typename T>
    class IntrusivePtr {
    public:

    	using NonConstType = typename std::remove_const<T>::type;

        IntrusivePtr() : m_p(nullptr) { }

        // Explicit constructor from T* , because we want to avoid that this constructor can be used to convert implicitly to IntrusivePtr
        // somewhere in the code which then deletes the resource unexpectetly!
        // In this constructor/destructors we need a static_cast to really be sure if the type T  inherits somehow from ReferenceCounting<T>
        // This is done by matching addRefIntrusivPtr and releaseIntrusivePtr
        explicit IntrusivePtr(T* p) : m_p(p) {
            if(p) addRefIntrusivPtr<T>(m_p);
        }

        IntrusivePtr(const IntrusivePtr & rhs) : m_p(rhs.m_p) {
            if(m_p) addRefIntrusivPtr<T>(m_p);
        }

        // Move support (temporaries)
        // Copy construct from temporary
        IntrusivePtr(IntrusivePtr && rhs) : m_p( rhs.m_p ){
	        rhs.m_p = 0; // temporary will not invoke reference count because pointer is zero!
	    }

        ~IntrusivePtr() {
            if(m_p) releaseIntrusivePtr<T,true>(m_p);
        }


        // We want to assign the intrusive ptr to this class
        // m_p points to A, rhs->m_p  points to B
        // This means, decrease ref count of current object A, an set m_p=rhs->m_p
        // and increase ref count of rhs resource. This can by:
        // Copy and swap idiom, call by value to copy the IntrusivePtr (ref count of B increments)
        // swap this resource pointer into the local temporary rhs (only pointer swap)
        // deleting the local rhs at end of function decrements ref count of initial resource A
        IntrusivePtr& operator=(IntrusivePtr const & rhs) {
            IntrusivePtr(rhs).swap(*this); // swaps the resource pointers
            return *this; // delete rhs-> which decrements correctly our initial resource A!
        }

        // Move Assignment (from temporary)
		// Make sure rhs.m_p is zero and as a consequence the destruction of rhs does not invoke release!
	    IntrusivePtr & operator=(IntrusivePtr && rhs){
	        IntrusivePtr( std::move( rhs ) ).swap(*this);
	        return *this;
	    }

		// Reset the IntrusivePtr to some other resource B,
		// meaning decrementing our resource A and setting the new pointer to B
		// and incrementing B
		// Can also take a nullptr!, making it not default argument because avoiding syntax mistakes with release()
		// which does a complete different thing (takes off the IntrusivePtr)
        void reset(T* p) {
            // Make temporary intrusive pointer for *p (incrementing ref count B)
            // swapping pointers with our resource A, and deleting temporary, decrement A
            IntrusivePtr(p).swap(*this);
        }

        // Release a IntrusivePtr from managing the shared resource
        // Decrements ref count of this resource A but without deleting it!
        T* release() {
        	releaseIntrusivePtr<T,false>(m_p);
            T* p = m_p;
            m_p = nullptr;
            return p;
        }

		// Get the underlying pointer
        T* get() const { return m_p; }

        // Implicit cast to T*
        operator T*() const { return m_p; }
        // Implicit cast to T&
        operator T&() const { return *m_p; }

        T* operator->() const { return m_p; }

        void swap(IntrusivePtr& rhs) {
            std::swap(m_p, rhs.m_p);
        }
    private:
        T* m_p;
    };

};
#endif
