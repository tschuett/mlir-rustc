#pragma once

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
};

} // namespace rust_compiler

// /// Helper conversion for a Toy AST location to an MLIR location.
//  mlir::Location loc(const Location &loc) {
//    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
//    loc.line,
//                                     loc.col);
//  }
