#include <iostream>
#include <random>
#include <memory>
#include <Eigen/Dense>

#include "CudaFramework/CudaModern/CudaRefcounting.hpp"
#include "CudaFramework/General/CPUTimer.hpp"


int global;

class A : public utilCuda::ReferenceCounting<void,A> {
public:
    A(int i): a(i) {}
    ~A() {
        //std::cout << "delete A" << std::endl;
        ++global;
    };
    int foo() {
        return a;
    }
    int a;
};

class B {
public:
    B(int i): a(i) {}
    ~B() {
        //std::cout << "delete B" << std::endl;
        ++global;
    };
    int foo() {
        return a;
    }
    int a;
};

// Add the virtual refocounting stuff

namespace utilCudaVirtual {


class NonCopyable {
protected:
    NonCopyable() {}
    ~NonCopyable() {}
private:
    NonCopyable(const NonCopyable&) { }
    const NonCopyable& operator=(const NonCopyable&) {
        return *this;
    }
};

class ReferenceCounting : public NonCopyable {
public:
    ReferenceCounting() : m_ref(0) { }
    virtual ~ReferenceCounting() { }

    virtual long addRef() {
        return ++m_ref;
    }
    virtual void release() {
        if(!--m_ref) delete this;
    }

private:
    unsigned long int m_ref;
};

inline long intrusive_ptr_add_ref(ReferenceCounting* base) {
    return base->addRef();
}

inline void intrusive_ptr_release(ReferenceCounting* base) {
    base->release();
}

template<typename T>
class IntrusivePtr {
public:
    IntrusivePtr() : m_p(nullptr) { }
    explicit IntrusivePtr(T* p) : m_p(p) {
        if(p) intrusive_ptr_add_ref(p);
    }
    IntrusivePtr(const IntrusivePtr<T>& rhs) : m_p(rhs.m_p) {
        if(m_p) intrusive_ptr_add_ref(m_p);
    }
    ~IntrusivePtr() {
        if(m_p) intrusive_ptr_release(m_p);
    }
    IntrusivePtr& operator=(const IntrusivePtr& rhs) {
        IntrusivePtr(rhs.get()).swap(*this);
        return *this;
    }

    void reset(T* p = 0) {
        IntrusivePtr(p).swap(*this);
    }
    T* release() {
        T* p = m_p;
        m_p = 0;
        return p;
    }

    T* get() const {
        return m_p;
    }
    operator T*() const {
        return m_p;    // why is this allowed IntrusivePtr<>()[3] shifts T* 3 times??
    }
    T* operator->() const {
        return m_p;
    }

    void swap(IntrusivePtr& rhs) {
        std::swap(m_p, rhs.m_p);
    }
private:
    T* m_p;
};

};

class C : public utilCudaVirtual::ReferenceCounting {
public:
    C(int i): a(i) {}
    virtual ~C() {
        //std::cout << "delete C" << std::endl;
       global++;
    };
    int foo() {
        return a;
    }
    int a;
};


int main() {


    int loops = 100;

    std::cout << "Raw Pointers Test" << std::endl;
    {
        int c = 0;
        global = 0;
        using namespace utilCuda;

        // Make vector of pointers
        std::vector<A * > vec;
        for(int i = 0; i< 10000000; i++) {
            vec.push_back( new A(i) );

        }
        START_TIMER(start)
        for(int i=0; i<loops; i++) {
            for(auto & p : vec) {
                c += p->foo();
            }
        }
        STOP_TIMER_MILLI(time,start);
        vec.clear();
        std::cout <<"Raw Pointers  AccessLoop: [s] " << time << " value: " << c << " deleted:" << global << std::endl;
    }


    std::cout << "IntrusivPtr Test" << std::endl;
    {
        int c = 0;
        global = 0;
        using namespace utilCuda;

        // Make vector of pointers
        std::vector<IntrusivePtr<A> > vec;
        for(int i = 0; i< 10000000; i++) {
            vec.push_back( IntrusivePtr<A>(new A(i))    );
        }
        START_TIMER(start)
        for(int i=0; i<loops; i++) {
            for(auto & p : vec) {
                c += p->foo();
            }
        }
        STOP_TIMER_MILLI(time,start);
        vec.clear();
        std::cout <<"IntrusivePtr  AccessLoop: [s] " << time << " value: " << c << " deleted:" << global << std::endl;
    }

    std::cout << "IntrusivPtr [virtual dispatch] Test" << std::endl;
    {
        int c = 0;
        global = 0;
        using namespace utilCudaVirtual;

        // Make vector of pointers
        std::vector<IntrusivePtr<C> > vec;
        for(int i = 0; i< 10000000; i++) {
            vec.push_back( IntrusivePtr<C>(new C(i))    );
        }
        START_TIMER(start)
        for(int i=0; i<loops; i++) {
            for(auto & p : vec) {
                c += p->foo();
            }
        }
        STOP_TIMER_MILLI(time,start);
        vec.clear();
        std::cout <<"IntrusivePtr  AccessLoop: [s] " << time << " value: " << c << " deleted:" << global << std::endl;
    }



    std::cout << "std::shared_ptr (make_shared) Test" << std::endl;
    {
        int c = 0;
        global = 0;
        // Make vector of pointers
        std::vector<std::shared_ptr<B> > vec;
        for(int i = 0; i< 10000000; i++) {
            vec.push_back( std::make_shared<B>(i) );
        }
        START_TIMER(start)
        for(int i=0; i<loops; i++) {
            for(auto & p : vec) {
                c += p->foo();
            }
        }
        STOP_TIMER_MILLI(time,start);
        vec.clear();
        std::cout <<"std::shared_ptr AccessLoop: [s] " << time << " value: " << c << " deleted:" << global << std::endl;
    }

     std::cout << "std::shared_ptr Test" << std::endl;
    {
        int c = 0;
        global = 0;
        // Make vector of pointers
        std::vector<std::shared_ptr<B> > vec;
        for(int i = 0; i< 10000000; i++) {
            vec.push_back( std::shared_ptr<B>( new B(i))    );
        }
        START_TIMER(start)
        for(int i=0; i<loops; i++) {
            for(auto & p : vec) {
                c += p->foo();
            }
        }
        STOP_TIMER_MILLI(time,start);
        vec.clear();
        std::cout <<"std::shared_ptr AccessLoop: [s] " << time << " value: " << c << " deleted:" << global << std::endl;
    }

};
