#pragma once

#include "AST/AST.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

class Abi : public Node {
  std::string abi;

public:
  Abi(Location loc) : Node(loc) {}

  void setString(std::string_view abi_) { abi = abi_; }

  std::string getAbi() const { return abi; }
};

} // namespace rust_compiler::ast
