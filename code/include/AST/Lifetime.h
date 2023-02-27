
#pragma once

#include "AST/AST.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

class Lifetime : public Node {
  std::string lifeTime;

public:
  Lifetime(Location loc) : Node(loc) {}

  void setLifetime(std::string_view s) { lifeTime = s; }
};

} // namespace rust_compiler::ast
