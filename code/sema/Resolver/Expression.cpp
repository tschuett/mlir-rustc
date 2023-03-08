#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveExpression(std::shared_ptr<ast::Expression> expr,
                                 const CanonicalPath &prefix,
                                 const CanonicalPath &canonicalPrefix) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    resolveExpressionWithBlock(
        std::static_pointer_cast<ast::ExpressionWithBlock>(expr), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    resolveExpressionWithoutBlock(
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(expr), prefix,
        canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  // FIXME
}

void Resolver::resolveExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock>,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  // FIXME
}

} // namespace rust_compiler::sema::resolver
