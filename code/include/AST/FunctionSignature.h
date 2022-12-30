#pragma once

#include "AST/AST.h"
#include "AST/Type.h"

#include <span>
#include <vector>

namespace rust_compiler::ast {

class Argument {
  std::shared_ptr<Type> type;

public:
  Type getType();
};

class FunctionSignature : public Node {
  std::vector<Argument> args;

public:
  std::string getName();

  std::span<Argument> getArgs() { return args; }

  std::shared_ptr<Type> getResult();
};

} // namespace rust_compiler::ast
