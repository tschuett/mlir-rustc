#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

bool Sema::isReprAttribute(const SimplePath &path) const {
  return path.getNrOfSegments() == 1 && !path.getSegment(0).isKeyWord() &&
    path.getSegment(0).getName() == Identifier("repr");
}

} // namespace rust_compiler::sema
