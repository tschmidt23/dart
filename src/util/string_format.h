#ifndef STRING_FORMAT_H
#define STRING_FORMAT_H

#include <string>

namespace dart {

/**
 * @namespace dart
 * This function mimics the behavior of sprintf, but returns a std::string rather than printing to a buffer. This can be useful if, for example, the size of the string after the format is unknown a priori.
 * @param fmt The format, using the same semantics as the standard printf.
 * @return The formatted string.
 */
std::string stringFormat(const std::string fmt, ...);

}

#endif // STRING_FORMAT_H
