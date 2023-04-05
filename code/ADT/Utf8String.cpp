#include "ADT/Utf8String.h"

#include <unicode/uchar.h>
#include <unicode/ucnv.h>
#include <unicode/utext.h>

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::adt {

bool Utf8String::isASCII() const {
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
  bool isASCII = true;

  for (UChar32 c : storage)
    if (c > 127)
      isASCII = false;

  return isASCII;
}

std::string Utf8String::toString() const {
  //  assert(false);
  //  UErrorCode status = U_ZERO_ERROR;
  //
  //  UConverter *conv = ucnv_open("US-ASCII", &status);
  //
  //  ucnv_close(conv);

  std::string result;
  for (UChar32 c : storage) {
    if (c > 127)
      assert(false && "no ascii");
    result.push_back(c);
  }

  //llvm::errs() << "Utf8String::toString(): " << storage.size() << "\n";

  return result;
}

void Utf8String::clear() { storage.clear(); }

void Utf8String::push_back(char c) { storage.push_back(c); }

} // namespace rust_compiler::adt
