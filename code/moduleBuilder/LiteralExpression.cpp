#include "AST/LiteralExpression.h"

#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <_types/_uint64_t.h>
#include <cstdlib>

using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitLiteralExpression(
    std::shared_ptr<ast::LiteralExpression> lit) {
  switch (lit->getLiteralKind()) {
  case LiteralExpressionKind::CharLiteral: {
    break;
  }
  case LiteralExpressionKind::StringLiteral: {
    break;
  }
  case LiteralExpressionKind::RawStringLiteral: {
    break;
  }
  case LiteralExpressionKind::ByteLiteral: {
    break;
  }
  case LiteralExpressionKind::ByteStringLiteral: {
    break;
  }
  case LiteralExpressionKind::RawByteStringLiteral: {
    break;
  }
  case LiteralExpressionKind::IntegerLiteral: {
    std::string value = lit->getValue();
    uint64_t integer = stoi(value);
    std::shared_ptr<ast::types::Type> type = lit->getType();
    return builder.create<Mir::ConstantOp>(
        getLocation(lit->getLocation()),
        builder.getIntegerAttr(builder.getI64Type(), integer),
        builder.getI64Type());
    break;
  }
  case LiteralExpressionKind::FloatLiteral: {
    break;
  }
  case LiteralExpressionKind::True: {
    return builder.create<Mir::ConstantOp>(
        getLocation(lit->getLocation()),
        builder.getIntegerAttr(builder.getI1Type(), 1), builder.getI1Type());
    break;
  }
  case LiteralExpressionKind::False: {
    return builder.create<Mir::ConstantOp>(
        getLocation(lit->getLocation()),
        builder.getIntegerAttr(builder.getI1Type(), 0), builder.getI1Type());
    break;
  }
  }

  llvm::outs() << "emitLiteralExpression: failed"
               << "\n";

  exit(EXIT_FAILURE);

  return nullptr;
}

} // namespace rust_compiler
