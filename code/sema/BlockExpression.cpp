#include "AST/Decls.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/Statement.h"
#include "Sema/Sema.h"

#include <memory>

namespace rust_compiler::sema {

void Sema::analyzeStatements(std::shared_ptr<ast::Statements> stmts) {
  for (auto &stmt : stmts->getStmts()) {
    switch (stmt->getKind()) {
    case ast::StatementKind::ItemDeclaration: {
      analyzeItemDeclaration(std::static_pointer_cast<ast::Item>(stmt));
      return;
    }
    case ast::StatementKind::LetStatement: {
      analyzeLetStatement(std::static_pointer_cast<ast::LetStatement>(stmt));
      return;
    }
    case ast::StatementKind::ExpressionStatement: {
      analyzeExpressionStatement(
          std::static_pointer_cast<ast::ExpressionStatement>(stmt));
      return;
    }
    case ast::StatementKind::MacroInvocationSemi: {
      analyzeMacroInvocation(
          std::static_pointer_cast<ast::MacroInvocation>(stmt));
      return;
    }
    }
  }
}

void Sema::analyzeBlockExpression(std::shared_ptr<ast::BlockExpression> block) {
  analyzeStatements(block->getExpressions());
}

} // namespace rust_compiler::sema
