#include "AST/Visiblity.h"

#include <cassert>

namespace rust_compiler::ast {

size_t Visibility::getTokens() {

  switch (kind) {
  case VisibilityKind::Public:
    return 1;
  case VisibilityKind::Private:
    return 0;
  case VisibilityKind::PublicCrate:
    return 4;
  case VisibilityKind::PublicSelf:
    return 4;
  case VisibilityKind::PublicSuper:
    return 4;
  case VisibilityKind::PublicIn: {
    return 4 + simplePath.getTokens();
  }
  }
}

} // namespace rust_compiler::ast
