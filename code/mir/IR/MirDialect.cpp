#include "Mir/MirDialect.h"
#include "Mir/MirOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

#include <llvm/Support/Debug.h>
#include <llvm/Support/WithColor.h>

#define DEBUG_TYPE "MirDialect"

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

//#include "Mir/MirOps.cpp.inc"

namespace rust_compiler::Mir {


} // namespace rust_compiler::Mir
