#pragma once

#include <mlir/IR/BuiltinAttributes.h>

namespace rust_compiler::mir {
}

#include "MirEnumAttr.h.inc"

#define GET_ATTRDEF_CLASSES
#include "MirAttr.h.inc"
