#include "AST/LetStatement.h"

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::crate_builder {

class Variable {};

[[maybe_unused]]static std::vector<Variable>
getVariables(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat) {
  assert(false);
  switch (pat->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    std::shared_ptr<ast::patterns::PatternWithoutRange> pattern =
        std::static_pointer_cast<PatternWithoutRange>(pat);
    switch (pattern->getWithoutRangeKind()) {
    case PatternWithoutRangeKind::LiteralPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::IdentifierPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::WildcardPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::RestPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::ReferencePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::StructPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::TupleStructPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::TuplePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::GroupedPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::SlicePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::PathPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::MacroInvocation: {
      assert(false);
    }
    }
  }
  case PatternNoTopAltKind::RangePattern: {
  }
  }
}

void CrateBuilder::emitLetStatement(ast::LetStatement *stmt) {

  assert(false);

  // memrefType

  assert(stmt->hasType());

  mlir::Type elementType = getType(stmt->getType().get());

  mlir::MemRefType memRef = mlir::MemRefType::Builder(1, elementType);

  [[maybe_unused]] mlir::Value addr = builder.create<mlir::memref::AllocaOp>(
      getLocation(stmt->getLocation()), memRef);

  allocaTable.insert(stmt->getPattern()->getNodeId(), addr);
}

} // namespace rust_compiler::crate_builder
