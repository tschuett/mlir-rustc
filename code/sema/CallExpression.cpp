#include "AST/Expression.h"
#include "AST/PathExpression.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeCallExpression(std::shared_ptr<ast::CallExpression> let) {
  std::shared_ptr<Expression> fun = let->getFunction();

  switch (fun->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    std::shared_ptr<ExpressionWithoutBlock> woBlock =
        std::static_pointer_cast<ExpressionWithoutBlock>(fun);
    switch (woBlock->getWithoutBlockKind()) {
    case ExpressionWithoutBlockKind::LiteralExpression: {
      break;
    }
    case ExpressionWithoutBlockKind::PathExpression: {
      std::shared_ptr<PathExpression> path =
          std::static_pointer_cast<PathExpression>(woBlock);
      break;
    }
    }
  }
  }
}

} // namespace rust_compiler::sema
