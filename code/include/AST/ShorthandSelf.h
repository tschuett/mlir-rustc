#pragma once

#include "AST/AST.h"
#include "AST/SelfParam.h"
#include "Location.h"

namespace rust_compiler::ast {

class ShorthandSelf : public SelfParam {
  bool mut = false;
  bool andP = false;

public:
  ShorthandSelf(Location loc) : SelfParam(loc) {}

  void setMut();
  void setAnd();

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
