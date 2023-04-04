#include "Lexer/Identifier.h"

#include <unicode/ucnv.h>
#include <unicode/utext.h>
#include <unicode/utf8.h>

namespace rust_compiler::lexer {

bool Identifier::isASCII() const {
  return storage.isASCII();
  //  UText ut = UTEXT_INITIALIZER;
  //  UErrorCode status = U_ZERO_ERROR;
  //  utext_openUTF8(&ut, storage.c_str(), -1, &status);
  //
  //  UChar32 c;
  //
  //  bool isAscii = true;
  //  for (c = utext_next32From(&ut, 0); c >= 0; c = utext_next32(&ut)) {
  //    if (c > 127)
  //      isAscii = false;
  //  }
  //
  //  utext_close(&ut);
  //
  //  return isAscii;
}

bool Identifier::isRawIdentifier() const { assert(false); }

std::string Identifier::toString() const {
  assert(false);
  UErrorCode status = U_ZERO_ERROR;

  UConverter *conv = ucnv_open("US-ASCII", &status);

  ucnv_close(conv);
}

Identifier::Identifier(std::string_view id) {
  storage.clear();
  for (char c : id)
    storage.push_back(c);
}

} // namespace rust_compiler::lexer
