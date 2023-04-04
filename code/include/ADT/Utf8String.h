#pragma once

#include <string>
#include <unicode/uchar.h>
#include <vector>

namespace rust_compiler::adt {

class Utf8String {
  /// an array of UChar32
  std::vector<UChar32> storage;

public:
  void append(UChar32);

  bool isASCII() const;

  std::string toString() const;

  void clear();
  void push_back(char);
};

} // namespace rust_compiler::adt
