#pragma once

#include <string>

namespace rust_compiler::lexer {

class Location {
  std::string fileName;
  std::string lineNumber;
  std::string columnName;

public:
};

} // namespace rust_compiler::lexer

/// Helper conversion for a Toy AST location to an MLIR location.
//  mlir::Location loc(const Location &loc) {
//    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
//    loc.line,
//                                     loc.col);
//  }
