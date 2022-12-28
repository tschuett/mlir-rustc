#include "Mir/MirDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include <mlir/IR/Types.h>

using namespace mlir;
using namespace mlir::mir;

#include "Mir/MirDialect.cpp.inc"

void Mir::MirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mir/MirOps.cpp.inc"
      >();
}
