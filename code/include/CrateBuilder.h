#pragma once

#include <string_view>

namespace rust_compiler {

class AST;

class CrateBuilder {
public:
  CrateBuilder(std::string_view moduleName);

  void build(AST *);
};

} // namespace rust_compiler
