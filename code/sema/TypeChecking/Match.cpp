#include "TyTy.h"
#include "TypeChecking.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkMatchExpression(std::shared_ptr<ast::MatchExpression> m) {

  assert(false && "to be implemented");
  [[maybe_unused]]TyTy::BaseType *scrutineeType =
      checkExpression(m->getScrutinee().getExpression());

  MatchArms matchArms = m->getMatchArms();

  std::vector<std::pair<MatchArm, std::shared_ptr<ast::Expression>>> arms =
      matchArms.getArms();

  for (auto &arm : arms) {
    auto [ar, expr] = arm;
  }
}

} // namespace rust_compiler::sema::type_checking
