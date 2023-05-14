#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/MacroInvocationSemiStatement.h"
#include "AST/Statement.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeStatements(ast::Statements stmts) {
  for (auto &stmt : stmts.getStmts()) {
    switch (stmt->getKind()) {
    case StatementKind::EmptyStatement: {
      // empty;
      return;
    }
    case StatementKind::ItemDeclaration: {
      analyzeItemDeclaration(std::static_pointer_cast<Node>(stmt));
      return;
    }
    case StatementKind::LetStatement: {
      analyzeLetStatement(std::static_pointer_cast<LetStatement>(stmt));
      return;
    }
    case StatementKind::ExpressionStatement: {
      analyzeExpressionStatement(
          std::static_pointer_cast<ExpressionStatement>(stmt));
      return;
    }
    case StatementKind::MacroInvocationSemi: {
      analyzeMacroInvocationSemiStatement(
          std::static_pointer_cast<MacroInvocationSemiStatement>(stmt));
      return;
    }
    }
  }
}

void Sema::analyzeBlockExpression(ast::BlockExpression* block) {
  analyzeStatements(block->getExpressions());
}

} // namespace rust_compiler::sema
