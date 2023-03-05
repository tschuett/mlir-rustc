#pragma once

#include <mlir/IR/BuiltinAttributes.h>

namespace rust_compiler::lir {
}

#include "LirEnumAttr.h.inc"

#define GET_ATTRDEF_CLASSES
#include "LirAttr.h.inc"
