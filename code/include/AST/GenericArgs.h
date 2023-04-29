#pragma once

#include "AST/AST.h"
#include "AST/GenericArg.h"
#include "Location.h"

#include <vector>

namespace rust_compiler::ast {

class GenericArgs : public Node {
  std::vector<GenericArg> args;
  bool trailingComma = false;

public:
  GenericArgs(Location loc) : Node(loc) {}

  void addArg(const GenericArg &arg) { args.push_back(arg); }
  void setTrailingSemi() { trailingComma = true; }

  static GenericArgs empty() {
    return GenericArgs(Location::getEmptyLocation());
  }

  std::vector<GenericArg> getArgs() const { return args; }

  size_t getNumberOfArgs() const { return args.size(); }

  bool isEmpty() const { return args.size() == 0; }
};

} // namespace rust_compiler::ast
