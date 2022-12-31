#pragma once

#include "AST/AST.h"
#include "AST/Type.h"

#include <mlir/IR/Location.h>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class Argument {
  std::shared_ptr<Type> type;

public:
  std::shared_ptr<Type> getType();
};

class FunctionSignature {
  std::vector<Argument> args;
  mlir::Location location;
  std::string name;
  std::shared_ptr<Type> resultType;

public:
  std::string getName();

  std::span<Argument> getArgs() { return args; }

  std::shared_ptr<Type> getResult();

  mlir::Location getLocation();
};

} // namespace rust_compiler::ast
