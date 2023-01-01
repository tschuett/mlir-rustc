#pragma once

#include <mlir/IR/Location.h>

namespace rust_compiler {

class LocationAttr : public mlir::LocationAttr {
  std::string fileName;
  unsigned lineNumber;

public:
  LocationAttr(std::string_view fileName, unsigned lineNumber)
      : fileName(fileName), lineNumber(lineNumber) {}

  unsigned getLineNumber() const { return lineNumber; }
};

} // namespace rust_compiler
