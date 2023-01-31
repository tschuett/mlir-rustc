#pragma once

#include "Sema/Common.h"

#include <memory>

namespace rust_compiler::ast {
class Expression;
}

namespace rust_compiler::sema {
class Sema;
}

namespace rust_compiler::sema {

class TypeChecking {
  Sema *sema;

public:
  TypeChecking(Sema *sema) : sema(sema) {}

  void eqExpr(AstId, AstId);
  void eqType(AstId, AstId);
  void sub();
};

} // namespace rust_compiler::sema
