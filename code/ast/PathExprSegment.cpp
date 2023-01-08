#include "AST/PathExprSegment.h"

#include <cassert>

namespace rust_compiler::ast {

  size_t PathExprSegment::getTokens() {
    size_t count = 0;
    for (GenericArgs arg: generics)
      count += arg.getTokens();
    return 1 + count;
  }
}
