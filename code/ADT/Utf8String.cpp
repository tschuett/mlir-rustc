#include "ADT/Utf8String.h"

#include <unicode/utext.h>

namespace rust_compiler::adt {

bool Utf8String::isASCII() const {

  UText ut = UTEXT_INITIALIZER;
  UErrorCode status = U_ZERO_ERROR;
  utext_openUTF8(&ut, storage.c_str(), -1, &status);

  UChar32 c;

  bool isAscii = true;
  for (c = utext_next32From(&ut, 0); c >= 0; c = utext_next32(&ut)) {
    if (c > 127)
      isAscii = false;
  }

  utext_close(&ut);

  return isAscii;
}

} // namespace rust_compiler::adt
