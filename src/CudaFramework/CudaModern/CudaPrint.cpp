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
