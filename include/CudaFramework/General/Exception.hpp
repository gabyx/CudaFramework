// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_Exception_hpp
#define CudaFramework_General_Exception_hpp

#include <stdexcept>
#include <exception>
#include <string>
#include <sstream>


class Exception : public std::runtime_error {
public:
    Exception(const std::stringstream & ss): std::runtime_error(ss.str()){};
private:

};

#define THROWEXCEPTION( message ) { std::stringstream ___s___ ; ___s___ << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; throw Exception(___s___); }



#endif // Exception_hpp
