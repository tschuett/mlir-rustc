#pragma once

#include "AST/AST.h"
#include "AST/Type.h"

#include "Location.h"

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
  Location location;
  std::string name;
  std::shared_ptr<Type> resultType;

public:
  FunctionSignature(Location loc);
  std::string getName();

  std::span<Argument> getArgs() { return args; }

  std::shared_ptr<Type> getResult();

  Location getLocation();

  void setName(std::string_view name);
};

} // namespace rust_compiler::ast


// TODO extern Abi
