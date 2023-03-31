#pragma once

#include <limits>
#include <llvm/Support/FormatVariadic.h>
#include <string>

namespace rust_compiler {

class Location {
  std::string fileName;
  unsigned lineNumber;
  unsigned columnNumber;

public:
  Location(std::string_view fileName, unsigned lineNumber,
           unsigned columnNumber)
      : fileName(fileName), lineNumber(lineNumber), columnNumber(columnNumber) {
  }

  std::string_view getFileName() const { return fileName; }
  unsigned getLineNumber() const { return lineNumber; }
  unsigned getColumnNumber() const { return columnNumber; }

  static Location getBuiltinLocation() {
    return Location("builtins.cpp", std::numeric_limits<unsigned>::max(),
                    std::numeric_limits<unsigned>::max());
  }

  static Location getEmptyLocation() {
    return Location("empty.cpp", std::numeric_limits<unsigned>::max(),
                    std::numeric_limits<unsigned>::max());
  }

  std::string toString() const {
    return llvm::formatv("{2} {0}:{1}", lineNumber, columnNumber, fileName)
        .str();
  };
};

} // namespace rust_compiler

// /// Helper conversion for a Toy AST location to an MLIR location.
//  mlir::Location loc(const Location &loc) {
//    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
//    loc.line,
//                                     loc.col);
//  }
