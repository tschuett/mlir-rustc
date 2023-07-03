#pragma once

#include <string>
#include <unicode/uchar.h>
#include <vector>

namespace rust_compiler::adt {

class Utf8String {
  /// an array of UChar32
  std::vector<UChar32> storage;

  // size_t size = 0;

public:
  Utf8String() = default;

  void append(UChar32);

  bool isASCII() const;

  std::string toString() const;

  void clear();

  size_t getLength() const { return storage.size(); }

  bool isEqualASCII(std::string_view ascii) const {
    if (storage.size() != ascii.size())
      return false;
    for (unsigned i = 0; i < storage.size(); ++i)
      if (storage[i] != ascii[i])
        return false;
    return true;
  }

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

  Utf8String &operator+=(const Utf8String &rhs) {
    for (size_t i = 0; i < rhs.storage.size(); ++i)
      storage.push_back(rhs.storage[i]);
    return *this;
  }
};

} // namespace rust_compiler::adt
