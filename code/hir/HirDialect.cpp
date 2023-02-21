#include "Hir/HirDialect.h"

#include "Hir/HirEnum.h"
#include "Hir/HirInterfaces.h"
#include "Hir/HirOps.h"
#include "Hir/HirString.h"
#include "Hir/HirStruct.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/WithColor.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

using namespace mlir;

#include "Hir/HirDialect.cpp.inc"

namespace rust_compiler::hir {

void HirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hir/HirOps.cpp.inc"
      >();
  //  addInterfaces<MirInlinerInterface>();
  addTypes<StructType, EnumType, StringType>();
}

} // namespace rust_compiler::hir
