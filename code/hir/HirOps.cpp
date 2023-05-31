#include "Hir/HirOps.h"

#include "Hir/HirDialect.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#define GET_OP_CLASSES
#include "Hir/HirOps.cpp.inc"

using namespace rust_compiler::hir;


namespace rust_compiler::hir {} // namespace rust_compiler::hir
