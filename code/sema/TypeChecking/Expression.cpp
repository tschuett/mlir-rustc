#include "AST/Expression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayElements.h"
#include "AST/AssignmentExpression.h"
#include "AST/BlockExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/ExpressionStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/Scrutinee.h"
#include "AST/Statement.h"
#include "Basic/Ids.h"
#include "Casting.h"
#include "Coercion.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "PathProbing.h"
#include "Sema/Autoderef.h"
#include "Sema/Properties.h"
#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"
#include "Unification.h"

#include "../ReturnExpressionSearcher.h"

#include <cassert>
#include <llvm/Support/ErrorHandling.h>
#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkExpression(std::shared_ptr<ast::Expression> expr) {
  TyTy::BaseType *infered = nullptr;
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    infered = checkExpressionWithBlock(
        std::static_pointer_cast<ast::ExpressionWithBlock>(expr));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    infered = checkExpressionWithoutBlock(
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(expr));
    break;
  }
  }
  infered->setReference(expr->getNodeId());
  tcx->insertType(expr->getIdentity(), infered);

  return infered;
}

TyTy::BaseType *TypeResolver::checkExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock> withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ast::ExpressionWithBlockKind::BlockExpression: {
    return checkBlockExpression(
        std::static_pointer_cast<ast::BlockExpression>(withBlock));
  }
  case ast::ExpressionWithBlockKind::UnsafeBlockExpression: {
    return checkUnsafeBlockExpression(
        std::static_pointer_cast<ast::UnsafeBlockExpression>(withBlock));
  }
  case ast::ExpressionWithBlockKind::LoopExpression: {
    assert(false && "to be implemented");
  }
  case ast::ExpressionWithBlockKind::IfExpression: {
    return checkIfExpression(std::static_pointer_cast<IfExpression>(withBlock));
  }
  case ast::ExpressionWithBlockKind::IfLetExpression: {
    assert(false && "to be implemented");
    return checkIfLetExpression(
        std::static_pointer_cast<IfLetExpression>(withBlock));
  }
  case ast::ExpressionWithBlockKind::MatchExpression: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    return checkLiteral(std::static_pointer_cast<LiteralExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    return checkPathExpression(
        std::static_pointer_cast<PathExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    return checkOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    return checkArrayExpression(
        std::static_pointer_cast<ArrayExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    return checkIndexExpression(
        std::static_pointer_cast<IndexExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    return checkCallExpression(static_cast<CallExpression *>(woBlock.get()));
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    return checkClosureExpression(
        std::static_pointer_cast<ClosureExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    return checkReturnExpression(
        std::static_pointer_cast<ReturnExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkBlockExpression(
    std::shared_ptr<ast::BlockExpression> block) {
  // assert(false && "to be implemented");
  Statements stmts = block->getExpressions();

  for (auto &s : stmts.getStmts()) {
    if (s->getKind() == StatementKind::ItemDeclaration)
      continue;

    checkStatement(s);
  }

  for (auto &s : stmts.getStmts()) {
    if (s->getKind() == StatementKind::ItemDeclaration)
      continue;

    TyTy::BaseType *stmtType = checkStatement(s);
    if (!stmtType) {
      llvm::errs() << "failed to resolve type: " << s->getLocation().toString()
                   << "\n";
      // report error
      return new TyTy::ErrorType(block->getNodeId());
    }

    if (s->getKind() == StatementKind::ExpressionStatement) {
      std::shared_ptr<ExpressionStatement> es =
          std::static_pointer_cast<ExpressionStatement>(s);
      if (es->getKind() == ExpressionStatementKind::ExpressionWithBlock) {
        // FIXME unify
      }
    }
  }

  if (stmts.hasTrailing())
    return checkExpression(stmts.getTrailing());
  else if (containsReturnExpression(block.get()))
    return new TyTy::NeverType(block->getNodeId());

  // FIXME
  return TyTy::TupleType::getUnitType(block->getNodeId());
}

TyTy::BaseType *TypeResolver::checkOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> op) {
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    return checkBorrowExpression(
        std::static_pointer_cast<BorrowExpression>(op));
  }
  case OperatorExpressionKind::DereferenceExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::NegationExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return checkArithmeticOrLogicalExpression(
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op));
  }
  case OperatorExpressionKind::ComparisonExpression: {
    return checkComparisonExpression(
        std::static_pointer_cast<ComparisonExpression>(op));
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::TypeCastExpression: {
    return checkTypeCastExpression(
        std::static_pointer_cast<TypeCastExpression>(op));
  }
  case OperatorExpressionKind::AssignmentExpression: {
    return checkAssignmentExpression(
        std::static_pointer_cast<AssignmentExpression>(op));
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith) {
  TyTy::BaseType *lhs = checkExpression(arith->getLHS());
  TyTy::BaseType *rhs = checkExpression(arith->getRHS());

  assert(lhs->getKind() != TyTy::TypeKind::Error);
  assert(rhs->getKind() != TyTy::TypeKind::Error);

  // FIXME resolveIOperatorOverload
  bool operatorOverloaded =
      resolveOperatorOverload(arith->getKind(), arith, lhs, rhs);
  assert(operatorOverloaded);

  if (!(validateArithmeticType(arith->getKind(), lhs) and
        validateArithmeticType(arith->getKind(), rhs))) {
    // report error
    llvm::errs() << arith->getLocation().toString()
                 << "cannot apply this operator to the given types"
                 << "\n";
    return new TyTy::ErrorType(arith->getNodeId());
  }

  switch (arith->getKind()) {
  case ArithmeticOrLogicalExpressionKind::LeftShift:
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    assert(false && "to be implemented");
  }
  default: {
    return Unification::unifyWithSite(
        TyTy::WithLocation(lhs, arith->getLHS()->getLocation()),
        TyTy::WithLocation(lhs, arith->getLHS()->getLocation()),
        arith->getLocation(), tcx);
  }
  }
}

TyTy::BaseType *TypeResolver::checkReturnExpression(
    std::shared_ptr<ast::ReturnExpression> ret) {
  Location loc = ret->hasTailExpression() ? ret->getExpression()->getLocation()
                                          : ret->getLocation();

  TyTy::BaseType *functionReturnTye = peekReturnType();

  assert(functionReturnTye != nullptr);

  TyTy::BaseType *ty = nullptr;
  if (ret->hasTailExpression()) {
    ty = checkExpression(ret->getExpression());
  } else {
    ty = TyTy::TupleType::getUnitType(ret->getNodeId());
  }

  [[maybe_unused]] TyTy::BaseType *infered = Unification::unifyWithSite(
      TyTy::WithLocation(functionReturnTye), TyTy::WithLocation(ty, loc),
      ret->getLocation(), tcx);

  return new TyTy::NeverType(ret->getNodeId());
}

bool TypeResolver::validateArithmeticType(
    ArithmeticOrLogicalExpressionKind kind, TyTy::BaseType *t) {
  // https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
  // reconsider with trait support
  switch (kind) {
  case ArithmeticOrLogicalExpressionKind::Addition:
  case ArithmeticOrLogicalExpressionKind::Subtraction:
  case ArithmeticOrLogicalExpressionKind::Multiplication:
  case ArithmeticOrLogicalExpressionKind::Division:
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    // integer or float
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::Float:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
      return true;
    case TyTy::TypeKind::Inferred:
      return (static_cast<TyTy::InferType *>(t)->getInferredKind() ==
              TyTy::InferKind::Integral) or
             (static_cast<TyTy::InferType *>(t)->getInferredKind() ==
              TyTy::InferKind::Float);
    default:
      return false;
    }
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd:
  case ArithmeticOrLogicalExpressionKind::BitwiseOr:
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    // integer or bools
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
    case TyTy::TypeKind::Bool:
      return true;
    case TyTy::TypeKind::Inferred:
      return static_cast<TyTy::InferType *>(t)->getInferredKind() ==
             TyTy::InferKind::Integral;
    default:
      return false;
    }
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift:
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    // integers
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
      return true;
    case TyTy::TypeKind::Inferred:
      return static_cast<TyTy::InferType *>(t)->getInferredKind() ==
             TyTy::InferKind::Integral;
    default:
      return false;
    }
  }
  }
}

bool TypeResolver::resolveOperatorOverload(
    ArithmeticOrLogicalExpressionKind,
    std::shared_ptr<ast::ArithmeticOrLogicalExpression>, TyTy::BaseType *,
    TyTy::BaseType *) {
  // assert(false);
  //  FIXME
  return true;
}

TyTy::BaseType *
TypeResolver::checkIfExpression(std::shared_ptr<ast::IfExpression> ifExpr) {
  checkExpression(ifExpr->getCondition());

  TyTy::BaseType *blk = checkExpression(ifExpr->getBlock());

  if (ifExpr->hasTrailing()) {
    TyTy::BaseType *elseBlk = checkExpression(ifExpr->getTrailing());

    if (blk->getKind() == TyTy::TypeKind::Never)
      return elseBlk;
    if (elseBlk->getKind() == TyTy::TypeKind::Never)
      return blk;

    return Unification::unifyWithSite(
        TyTy::WithLocation(blk, ifExpr->getBlock()->getLocation()),
        TyTy::WithLocation(elseBlk, ifExpr->getTrailing()->getLocation()),
        ifExpr->getLocation(), tcx);
  }

  return TyTy::TupleType::getUnitType(ifExpr->getNodeId());
}

TyTy::BaseType *TypeResolver::checkAssignmentExpression(
    std::shared_ptr<ast::AssignmentExpression> ass) {
  auto lhs = checkExpression(ass->getLHS());
  auto rhs = checkExpression(ass->getRHS());

  coercionWithSite(ass->getNodeId(),
                   TyTy::WithLocation(lhs, ass->getLHS()->getLocation()),
                   TyTy::WithLocation(rhs, ass->getRHS()->getLocation()),
                   ass->getLocation(), tcx);

  return TyTy::TupleType::getUnitType(ass->getNodeId());
}

TyTy::BaseType *TypeResolver::checkComparisonExpression(
    std::shared_ptr<ast::ComparisonExpression> cmp) {
  TyTy::BaseType *l = checkExpression(cmp->getLHS());
  TyTy::BaseType *r = checkExpression(cmp->getRHS());

  Unification::unifyWithSite(
      TyTy::WithLocation(l, cmp->getLHS()->getLocation()),
      TyTy::WithLocation(r, cmp->getRHS()->getLocation()), cmp->getLocation(),
      tcx);

  std::optional<TyTy::BaseType *> bo = tcx->lookupBuiltin("bool");
  if (bo)
    return *bo;

  assert(false);
}

TyTy::BaseType *TypeResolver::checkIfLetExpression(
    std::shared_ptr<ast::IfLetExpression> ifLet) {
  assert(false);

  Scrutinee scrut = ifLet->getScrutinee();
  TyTy::BaseType *scrutineeType = checkExpression(scrut.getExpression());

  std::shared_ptr<ast::patterns::Pattern> pattern = ifLet->getPatterns();

  for (auto pat : pattern->getPatterns()) {
    TyTy::BaseType *armType = checkPattern(pat, scrutineeType);

    Unification::unifyWithSite(TyTy::WithLocation(scrutineeType),
                               TyTy::WithLocation(armType, pat->getLocation()),
                               ifLet->getLocation(), tcx);
  }

  [[maybe_unused]] TyTy::BaseType *ifletBlock =
      checkExpression(ifLet->getIfLet());

  assert(false && "incomplete");
}

TyTy::BaseType *TypeResolver::checkCallExpression(CallExpression *call) {
  TyTy::BaseType *functionType = checkExpression(call->getFunction());

  TyTy::VariantDef variant = TyTy::VariantDef::getErrorNode();
  if (functionType->getKind() == TyTy::TypeKind::ADT) {
    TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(functionType);
    if (adt->isEnum()) {
      std::optional<basic::NodeId> variantId =
          tcx->lookupVariantDefinition(call->getFunction()->getNodeId());
      assert(variantId.has_value());

      TyTy::VariantDef *lookupVariant = nullptr;
      bool ok = adt->lookupVariantById(*variantId, &lookupVariant);
      assert(ok);
      variant = *lookupVariant;
    } else {
      assert(adt->getNumberOfVariants() == 1);
      variant = *adt->getVariants()[0];
    }

    return checkCallExpression(functionType, call, variant);
  }

  // handle traits
  std::optional<TyTy::BaseType *> trait =
      checkFunctionTraitCall(call, functionType);
  if (trait)
    return *trait;

  if ((functionType->getKind() == TyTy::TypeKind::Function) ||
      (functionType->getKind() == TyTy::TypeKind::FunctionPointer)) {
    return checkCallExpression(functionType, call, variant);
  } else {
    // report error
    assert(false);
  }
}

TyTy::BaseType *TypeResolver::checkCallExpression(TyTy::BaseType *functionType,
                                                  CallExpression *call,
                                                  TyTy::VariantDef &variant) {

  std::vector<TyTy::Argument> arguments;
  if (call->hasParameters()) {
    CallParams params = call->getParameters();
    for (auto &arg : params.getParams()) {
      TyTy::BaseType *argumentExpressionType = checkExpression(arg);
      if (argumentExpressionType->getKind() == TyTy::TypeKind::Error) {
        assert(false);
      }

      TyTy::Argument a = {arg->getIdentity(), argumentExpressionType,
                          arg->getLocation()};
      arguments.push_back(a);
    }
  }

  switch (functionType->getKind()) {
  case TyTy::TypeKind::ADT: {
    assert(false);
  }
  case TyTy::TypeKind::Function: {
    return checkCallExpressionFn(functionType, call, variant);
    break;
  }
  case TyTy::TypeKind::FunctionPointer: {
    assert(false);
  }
  case TyTy::TypeKind::Inferred:
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Error:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::Closure:
    return new TyTy::ErrorType(0);
  }
  llvm_unreachable("unknown type kind");
}

TyTy::BaseType *
TypeResolver::checkCallExpressionFn(TyTy::BaseType *functionType,
                                    ast::CallExpression *call,
                                    TyTy::VariantDef &variant) {
  TyTy::FunctionType *fun = static_cast<TyTy::FunctionType *>(functionType);

  // check number of paramters minus variadic
  if (call->getNumberOfParams() != fun->getNumberOfArguments()) {
    if (fun->isVaradic()) {
      if (call->getNumberOfParams() < fun->getNumberOfArguments()) {
        // report error
        assert(false);
      }
    } else {
      // report error
      assert(false);
    }
  }

  // call expressions and funtion types must match
  size_t i = 0;
  for (auto fp : call->getParameters().getParams()) {
    TyTy::BaseType *argExprType = checkExpression(fp);
    if (argExprType->getKind() == TyTy::TypeKind::Error) {
      // report error
      assert(false);
    }

    // variadic?
    if (i < fun->getNumberOfArguments()) {
      auto param = fun->getParameter(i);
      auto pattern = param.first;
      TyTy::BaseType *paramTy = param.second;
      basic::NodeId coercionSiteId = fp->getNodeId();

      // FIXME: add more locations
      TyTy::BaseType *resolvedArgumentType = coercionWithSite(
          coercionSiteId, TyTy::WithLocation(paramTy),
          TyTy::WithLocation(argExprType), fp->getLocation(), tcx);
      if (resolvedArgumentType->getKind() == TyTy::TypeKind::Error) {
        // report error
        assert(false);
      }
    }

    //    else { // no!
    //      assert(false);
    //      // FIXME: todo
    //      // switch(argExprType->getKind()) {
    //      // case TypeKind::Error : {
    //      //  assert(false);
    //      //}
    //      // case TypeKind::Int: {
    //      //}
    //      //}
    //    }
    ++i;
  }

  // check again
  if (i < call->getNumberOfParams()) {
    // report error
    assert(false);
  }

  // FIXME
  return fun->getReturnType()->clone();
}

std::optional<TyTy::BaseType *>
TypeResolver::checkFunctionTraitCall(CallExpression *expr,
                                     TyTy::BaseType *functionType) {
  TyTy::TypeBoundPredicate associatedPredicate =
      TyTy::TypeBoundPredicate::error();
  std::optional<TyTy::FunctionTrait> methodTrait =
      checkPossibleFunctionTraitCallMethodName(*functionType,
                                               &associatedPredicate);
  if (!methodTrait)
    return std::nullopt;

  std::set<MethodCandidate> candidates =
      resolveMethodProbe(functionType, *methodTrait);
  if (candidates.empty())
    return std::nullopt;

  if (candidates.size() > 1) {
    // report error
    assert(false);
  }

  if (functionType->getKind() == TyTy::TypeKind::Closure) {
    [[maybe_unused]] TyTy::ClosureType *clos =
        static_cast<TyTy::ClosureType *>(functionType);
    // FIXME: TODO
    // clos->setupFnOnceOutput();
  }

  MethodCandidate candidate = *candidates.begin();

  Adjuster adj(functionType);
  TyTy::BaseType *adjustedSelf = adj.adjustType(candidate.getAdjustments());

  tcx->insertAutoderefMapping(expr->getNodeId(), candidate.getAdjustments());
  tcx->insertReceiver(expr->getNodeId(), functionType);

  PathProbeCandidate resolvedCandidate = candidate.getCandidate();
  TyTy::BaseType *lookupType = resolvedCandidate.getType();
  basic::NodeId resolvedNodeId = resolvedCandidate.isImplCandidate()
                                     ? resolvedCandidate.getImplNodeId()
                                     : resolvedCandidate.getTraitNodeId();

  if (lookupType->getKind() != TyTy::TypeKind::Function) {
    // report error
    assert(false);
  }

  TyTy::FunctionType *fn = static_cast<TyTy::FunctionType *>(lookupType);
  if (!fn->isMethod()) {
    // report error
    assert(false);
  }

  std::vector<TyTy::TypeVariable> callArgs;
  if (expr->hasParameters()) {
    for (auto &p : expr->getParameters().getParams()) {
      TyTy::BaseType *a = checkExpression(p);
      callArgs.push_back(TyTy::TypeVariable(a->getReference()));
    }
  }

  basic::NodeId implicitArgId = basic::getNextNodeId();
  NodeIdentity identity = {
      implicitArgId, rust_compiler::session::session->getCurrentCrateNum(),
      Location::getEmptyLocation()};

  TyTy::TupleType *tuple =
      new TyTy::TupleType(implicitArgId, expr->getLocation(), callArgs);
  tcx->insertImplicitType(implicitArgId, tuple);

  std::vector<TyTy::Argument> args;
  TyTy::Argument a = {identity, tuple, expr->getLocation()};
  args.push_back(a);

  TyTy::BaseType *functionReturnType = checkMethodCallExpression(
      fn, expr->getIdentity(), args, expr->getLocation(), expr->getLocation(),
      adjustedSelf);

  if (functionReturnType == nullptr ||
      functionReturnType->getKind() == TyTy::TypeKind::Error) {
    // report error
    assert(false);
  }

  tcx->insertOperatorOverLoad(expr->getNodeId(), fn);
  tcx->insertResolvedName(expr->getNodeId(), resolvedNodeId);

  return functionReturnType;
}

std::optional<TyTy::FunctionTrait>
TypeResolver::checkPossibleFunctionTraitCallMethodName(
    TyTy::BaseType &receiver, TyTy::TypeBoundPredicate *associatedPredicate) {

  for (auto &bound : receiver.getSpecifiedBounds()) {
    if (bound.getIdentifier() == lexer::Identifier("Fn")) {
      *associatedPredicate = bound;
      return TyTy::FunctionTrait::Fn;
    }
    if (bound.getIdentifier() == lexer::Identifier("FnMut")) {
      *associatedPredicate = bound;
      return TyTy::FunctionTrait::FnMut;
    }
    if (bound.getIdentifier() == lexer::Identifier("FnOnce")) {
      *associatedPredicate = bound;
      return TyTy::FunctionTrait::FnOnce;
    }
  }

  return std::nullopt;
}

TyTy::BaseType *TypeResolver::checkMethodCallExpression(
    TyTy::FunctionType *, NodeIdentity, std::vector<TyTy::Argument> &args,
    Location call, Location receiver, TyTy::BaseType *adjustedSelf) {
  assert(false);
}

TyTy::BaseType *TypeResolver::checkUnsafeBlockExpression(
    std::shared_ptr<ast::UnsafeBlockExpression> unsafe) {
  return checkExpression(unsafe->getBlock());
}

TyTy::BaseType *
TypeResolver::checkArrayExpression(std::shared_ptr<ast::ArrayExpression> arr) {
  TyTy::BaseType *elementType = nullptr;
  std::shared_ptr<Expression> capacityExpr;
  if (arr->hasArrayElements()) {
    ArrayElements elements = arr->getArrayElements();
    switch (elements.getKind()) {
    case ArrayElementsKind::List: {
      std::vector<TyTy::BaseType *> types;
      for (auto &el : elements.getElements())
        types.push_back(checkExpression(el));

      elementType =
          TyTy::TypeVariable::getImplicitInferVariable(arr->getLocation())
              .getType();

      for (auto &type : types) {
        elementType = Unification::unifyWithSite(
            TyTy::WithLocation(elementType),
            TyTy::WithLocation(type, type->getLocation()), arr->getLocation(),
            tcx);
      }

      std::shared_ptr<LiteralExpression> lit =
          std::make_shared<LiteralExpression>(arr->getLocation());
      lit->setKind(LiteralExpressionKind::IntegerLiteral);
      lit->setStorage(std::to_string(elements.getNumberOfElements()));
      capacityExpr = lit;

      std::optional<TyTy::BaseType *> expectedType =
          tcx->lookupBuiltin("usize");
      assert(expectedType.has_value());
      tcx->insertType(
          NodeIdentity(basic::UNKNOWN_NODEID,
                       rust_compiler::session::session->getCurrentCrateNum(),
                       (*expectedType)->getLocation()),
          *expectedType);
      break;
    }
    case ArrayElementsKind::Repeated: {
      elementType = checkExpression(elements.getValue());
      TyTy::BaseType *capacityType = checkExpression(elements.getCount());

      std::optional<TyTy::BaseType *> expectedType =
          tcx->lookupBuiltin("usize");
      assert(expectedType.has_value());
      tcx->insertType(elements.getCount()->getIdentity(), *expectedType);

      Unification::unifyWithSite(
          TyTy::WithLocation(*expectedType),
          TyTy::WithLocation(capacityType, elements.getCount()->getLocation()),
          arr->getLocation(), tcx);

      capacityExpr = elements.getCount();
      break;
    }
    }
    return new TyTy::ArrayType(arr->getNodeId(), arr->getLocation(),
                               capacityExpr,
                               TyTy::TypeVariable(elementType->getReference()));
  }
  assert(false);
}

TyTy::BaseType *TypeResolver::checkTypeCastExpression(
    std::shared_ptr<ast::TypeCastExpression> cast) {
  TyTy::BaseType *left = checkExpression(cast->getLeft());
  TyTy::BaseType *right = checkType(cast->getRight());

  return Casting::castWithSite(
      cast->getNodeId(),
      TyTy::WithLocation(left, cast->getLeft()->getLocation()),
      TyTy::WithLocation(right, cast->getRight()->getLocation()),
      cast->getLocation());
}

TyTy::BaseType *TypeResolver::checkBorrowExpression(
    std::shared_ptr<ast::BorrowExpression> borrow) {
  TyTy::BaseType *base = checkExpression(borrow->getExpression());

  if (base->getKind() == TyTy::TypeKind::Reference) {
    TyTy::ReferenceType *ref = static_cast<TyTy::ReferenceType *>(base);

    if (ref->isDynStrType()) {
      return ref;
    }
  }

  return new TyTy::ReferenceType(borrow->getNodeId(),
                                 TyTy::TypeVariable(base->getReference()),
                                 borrow->getMutability());
}

TyTy::BaseType *
TypeResolver::checkIndexExpression(std::shared_ptr<IndexExpression> index) {
  TyTy::BaseType *arrayExprType = checkExpression(index->getArray());
  if (arrayExprType->getKind() == TyTy::TypeKind::Error) {
    assert(false);
  }

  TyTy::BaseType *indexExprType = checkExpression(index->getIndex());
  if (indexExprType->getKind() == TyTy::TypeKind::Error) {
    assert(false);
  }

  TyTy::BaseType *copyArrayExprType = arrayExprType;
  if (arrayExprType->getKind() == TyTy::TypeKind::Reference) {
    TyTy::ReferenceType *ref =
        static_cast<TyTy::ReferenceType *>(copyArrayExprType);
    TyTy::BaseType *base = ref->getBase();
    if (base->getKind() == TypeKind::Array) {
      copyArrayExprType = base;
    }
  }

  std::optional<TyTy::BaseType *> usize = tcx->lookupBuiltin("usize");
  assert(usize.has_value());
  if (indexExprType->canEqual(*usize, false) &&
      copyArrayExprType->getKind() == TypeKind::Array) {

    Unification::unifyWithSite(
        TyTy::WithLocation(*usize),
        TyTy::WithLocation(indexExprType, index->getIndex()->getLocation()),
        index->getLocation(), tcx);

    TyTy::ArrayType *arrayType =
        static_cast<TyTy::ArrayType *>(copyArrayExprType);
    return arrayType->getElementType()->clone();
  }

  std::optional<TyTy::BaseType *> overloaded = resolveOperatorOverload(
      PropertyKind::INDEX, index.get(), arrayExprType, indexExprType);

  if (overloaded) {

    TyTy::BaseType *resolved = *overloaded;
    assert(resolved->getKind() == TypeKind::Reference);
    TyTy::ReferenceType *ref = static_cast<TyTy::ReferenceType *>(resolved);

    return ref->getBase()->clone();
  }

  assert(false);
}

} // namespace rust_compiler::sema::type_checking
