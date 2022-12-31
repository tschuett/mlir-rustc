#include "Mir/MirDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include <mlir/IR/Types.h>

using namespace mlir;
using namespace rust_compiler::Mir;

#include "Mir/MirDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"

void MirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mir/MirOps.cpp.inc"
      >();
}



