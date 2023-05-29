#include "CrateBuilder/CrateBuilder.h"

#include "AST/Expression.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

// void build(std::shared_ptr<rust_compiler::ast::Crate> crate) {
//   mlir::MLIRContext context;
//
//   // FIXME: + .yaml
//   std::string fn = std::string(crate.getCrateName());
//
//   std::error_code EC;
//   llvm::raw_fd_stream stream = {fn, EC};
//
//   CrateBuilder builder = {stream, context};
//
//   builder->emitCrate(crate);
// }

void CrateBuilder::emitCrate(rust_compiler::ast::Crate *crate) {
  this->crate = crate;

  for (auto &i : crate->getItems()) {
    emitItem(i.get());
  }
}

bool CrateBuilder::isLiteralExpression(ast::Expression *expr) const {
  if (expr->getExpressionKind() == ExpressionKind::ExpressionWithoutBlock) {
    if (static_cast<ast::ExpressionWithoutBlock *>(expr)
            ->getWithoutBlockKind() ==
        ast::ExpressionWithoutBlockKind::LiteralExpression)
      return true;
    return false;
  }
  return false;
}

} // namespace rust_compiler::crate_builder
