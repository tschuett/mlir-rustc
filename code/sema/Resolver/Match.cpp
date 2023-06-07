#include "AST/MatchArm.h"
#include "AST/MatchArms.h"
#include "AST/Scrutinee.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "PatternDeclaration.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

void Resolver::resolveMatchExpression(
    std::shared_ptr<ast::MatchExpression> match,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(match->getScrutinee().getExpression(), prefix,
                    canonicalPrefix);
  for (auto &armAndExpr : match->getMatchArms().getArms()) {
    NodeId scopeNodeId = armAndExpr.second->getNodeId();
    getNameScope().push(scopeNodeId);
    getTypeScope().push(scopeNodeId);
    getLabelScope().push(scopeNodeId);
    pushNewNameRib(getNameScope().peek());
    pushNewTypeRib(getTypeScope().peek());
    pushNewLabelRib(getLabelScope().peek());

    MatchArm &arm = armAndExpr.first;
    if (arm.hasGuard())
      resolveExpression(arm.getGuard().getGuard(), prefix, canonicalPrefix);

    std::vector<PatternBinding> bindings = {
        PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

    for (auto &pattern : arm.getPattern()->getPatterns()) {
      PatternDeclaration decl = {pattern, RibKind::Variable, bindings, this,
                                 prefix,  canonicalPrefix};
      decl.resolve();
    }

    resolveExpression(armAndExpr.second, prefix, canonicalPrefix);

    getNameScope().pop();
    getTypeScope().pop();
    getLabelScope().pop();
  }
}

} // namespace rust_compiler::sema::resolver
