#include "AST/BlockExpression.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/Statement.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveStatement(std::shared_ptr<ast::Statement> stmt,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix,
                                const adt::CanonicalPath &enumPrefix) {
  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    assert(false && "to be handled later");
    break;
  }
  case StatementKind::ItemDeclaration: {
    resolveItemDeclaration(std::static_pointer_cast<ItemDeclaration>(stmt),
                           prefix, canonicalPrefix);
    break;
  }
  case StatementKind::LetStatement: {
    resolveLetStatement(std::static_pointer_cast<LetStatement>(stmt), prefix,
                        canonicalPrefix);
    break;
  }
  case StatementKind::ExpressionStatement: {
    resolveExpressionStatement(
        std::static_pointer_cast<ExpressionStatement>(stmt), prefix,
        canonicalPrefix);
    break;
  }
  case StatementKind::MacroInvocationSemi: {
    assert(false && "to be handled later");
    break;
  }
  }
}

void Resolver::resolveLetStatement(std::shared_ptr<ast::LetStatement> let,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  if (let->hasInit())
    resolveExpression(let->getInit(), prefix, canonicalPrefix);

  resolvePatternDeclaration(let->getPattern(), RibKind::Variable, prefix,
                            canonicalPrefix);

  if (let->hasType())
    resolveType(let->getType(), prefix, canonicalPrefix);

  if (let->hasElse())
    resolveBlockExpression(
        std::static_pointer_cast<BlockExpression>(let->getElse()), prefix,
        canonicalPrefix);
}

void Resolver::resolveExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> estmt,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (estmt->getKind()) {
  case ExpressionStatementKind::ExpressionWithBlock: {
    resolveExpression(estmt->getWithBlock(), prefix, canonicalPrefix);
    break;
  }
  case ExpressionStatementKind::ExpressionWithoutBlock: {
    resolveExpression(estmt->getWithoutBlock(), prefix, canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveStructStructStatement(
    std::shared_ptr<ast::StructStruct>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveTupleStructStatement(
    std::shared_ptr<ast::TupleStruct>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveItemDeclaration(
    std::shared_ptr<ast::ItemDeclaration> item,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (item->hasVisItem()) {
    resolveVisItem(item->getVisItem(), prefix, canonicalPrefix);
  } else if (item->hasMacroItem()) {
    assert(false);
  }
}

} // namespace rust_compiler::sema::resolver
