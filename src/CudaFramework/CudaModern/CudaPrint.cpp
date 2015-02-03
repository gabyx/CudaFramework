// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/CudaModern/CudaPrint.hpp"


std::string utilCuda::internal::stringprintf(const char* format, ...) {
	va_list args;
	va_start(args, format);
	int len = vsnprintf(0, 0, format, args);
	va_end(args);

	// allocate space.
	std::string text;
	text.resize(len);

	va_start(args, format);
	vsnprintf(&text[0], len + 1, format, args);
	va_end(args);

	return text;
}
