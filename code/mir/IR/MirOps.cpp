#include "Mir/MirOps.h"

#include "Mir/MirAttr.h"
#include "Mir/MirDialect.h"
#include "Mir/MirInterfaces.h"
#include "Mir/MirTypes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"

using namespace mlir;

namespace rust_compiler::Mir {

//===----------------------------------------------------------------------===//
// VTableOp
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//



} // namespace rust_compiler::Mir
