#pragma once

#include "Sema/Common.h"

#include <map>

namespace rust_compiler::ast {
  class Expression;
}

namespace rust_compiler::ast::types {
  class Type;
}

namespace rust_compiler::sema {

class Sema;

class Mappings {
  Sema *sema;

public:
  Mappings(Sema *sema) : sema(sema){};

private:
  };

} // namespace rust_compiler::sema
