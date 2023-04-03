#pragma once

#include <unicode/uchar.h>

#include <string>

namespace rust_compiler::adt {

class Utf8String {
  /// an array of UChar32
  std::string storage;

 public:
  void append(UChar32);

  bool isASCII() const;

  std::string toString() const;
};

} // namespace rust_compiler::adt
