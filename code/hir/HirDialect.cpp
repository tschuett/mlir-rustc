#include "Hir/HirDialect.h"

#include "Hir/HirTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
// #include <mlir/IR/ExtensibleDialect.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

using namespace mlir;

#include "Hir/HirDialect.cpp.inc"

using namespace rust_compiler::hir;

class HirInlinerInterface
    : public DialectInterface::Base<DialectInlinerInterface> {
public:
};

bool isScalarObject(mlir::Type type) {
  if (mlir::IntegerType integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return true;
  if (mlir::FloatType floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return true;
  return false;
}

bool isPattern(mlir::Type type) {
  return mlir::isa<rust_compiler::hir::PatternType>(type);
}

bool isRangePattern(mlir::Type type) {
  //  return mlir::isa<rust_compiler::hir::RangePatternType>(type);
}

bool isPatternWithoutRange(mlir::Type type) {
  return (mlir::isa<rust_compiler::hir::LiteralPatternType>(type) ||
          mlir::isa<rust_compiler::hir::IdentifierPatternType>(type) ||
          mlir::isa<rust_compiler::hir::WildcardPatternType>(type) ||
          mlir::isa<rust_compiler::hir::RestPatternType>(type) ||
          mlir::isa<rust_compiler::hir::ReferencePatternType>(type) ||
          mlir::isa<rust_compiler::hir::TupleStructPatternType>(type) ||
          mlir::isa<rust_compiler::hir::GroupedPatternType>(type) ||
          mlir::isa<rust_compiler::hir::SlicePatternType>(type) ||
          mlir::isa<rust_compiler::hir::PathPatternType>(type));
}

bool isPatternNoTopAlt(mlir::Type type) {
  return isPatternWithoutRange(type) || isRangePattern(type);
}

void HirDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "Hir/HirOps.cpp.inc"
      >();
}
