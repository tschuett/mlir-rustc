#pragma once

#include <string>
#include <unicode/uchar.h>
#include <vector>

namespace rust_compiler::adt {

class Utf8String {
  /// an array of UChar32
  std::vector<UChar32> storage;

  size_t size = 0;

public:
  Utf8String() = default;

  void append(UChar32);

  bool isASCII() const;

  std::string toString() const;

  void clear();
  void push_back(char);

  bool operator==(const Utf8String &b) const {
    if (storage.size() != b.storage.size())
      return false;
    for (unsigned i = 0; i < storage.size(); ++i)
      if (storage[i] != b.storage[i])
        return false;
    return true;
  }

  bool operator<(const Utf8String &b) const {
    if (storage.size() > b.storage.size())
      // if (size > b.size)
      return false;
    if (storage.size() < b.storage.size())
      return true;
    for (unsigned i = 0; i < storage.size(); ++i)
      if (storage[i] < b.storage[i])
        return true;
    return false;
  }
};

} // namespace rust_compiler::adt
