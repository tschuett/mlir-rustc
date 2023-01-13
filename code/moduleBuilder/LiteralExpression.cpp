#include "AST/LiteralExpression.h"

#include "ModuleBuilder/ModuleBuilder.h"

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
    std::shared_ptr<ast::types::Type> type = lit->getType();
    return builder.create<mlir::func::ConstantOp>(
        getLocation(lit->getLocation()), getType(type), xxx);
    break;
  }
  case LiteralExpressionKind::FloatLiteral: {
    break;
  }
  case LiteralExpressionKind::True: {
    break;
  }
  case LiteralExpressionKind::False: {
    break;
  }
  }

  llvm::outs() << "emitLiteralExpression: failed"
               << "\n";
  return nullptr;
}

} // namespace rust_compiler
