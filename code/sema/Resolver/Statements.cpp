#include "AST/Statement.h"
#include "Resolver.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveStatement(std::shared_ptr<ast::Statement> stmt,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix,
                                const adt::CanonicalPath &empty) {
  assert(false && "to be handled later");
  switch (stmt->getKind()) {
  case StatementKind::ItemDeclaration: {
    break;
  }
  case StatementKind::LetStatement: {
    break;
  }
  case StatementKind::ExpressionStatement: {
    break;
  }
  case StatementKind::MacroInvocationSemi: {
    break;
  }
  }
}

} // namespace rust_compiler::sema::resolver
