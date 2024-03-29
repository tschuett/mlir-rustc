#include "AST/Expression.h"

#include "ADT/CanonicalPath.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/BlockExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "AST/QualifiedPathInType.h"
#include "AST/RangeExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/Scrutinee.h"
#include "AST/Statements.h"
#include "AST/StructBase.h"
#include "AST/StructExprStruct.h"
#include "AST/StructExprUnit.h"
#include "AST/StructExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "AST/UnsafeBlockExpression.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "PatternDeclaration.h"
#include "Resolver.h"

#include <filesystem>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;

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
    std::shared_ptr<ast::ExpressionWithBlock> withBlock,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    resolveBlockExpression(std::static_pointer_cast<BlockExpression>(withBlock),
                           prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    resolveUnsafeBlockExpression(
        std::static_pointer_cast<UnsafeBlockExpression>(withBlock), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    resolveLoopExpression(std::static_pointer_cast<LoopExpression>(withBlock),
                          prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    resolveIfExpression(std::static_pointer_cast<IfExpression>(withBlock),
                        prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    resolveIfLetExpression(
        std::static_pointer_cast<IfLetExpression>(withBlock).get(), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    resolveMatchExpression(std::static_pointer_cast<MatchExpression>(withBlock),
                           prefix, canonicalPrefix);
  }
  }
}

void Resolver::resolveExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> woBlock,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    // ignored
    break;
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    resolvePathExpression(std::static_pointer_cast<PathExpression>(woBlock),
                          prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    resolveOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(woBlock), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    resolveArrayExpression(std::static_pointer_cast<ArrayExpression>(woBlock),
                           prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    resolveIndexExpression(std::static_pointer_cast<IndexExpression>(woBlock),
                           prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    resolveTupleExpression(
        std::static_pointer_cast<TupleExpression>(woBlock).get(), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    resolveTupleIndexingExpression(
        std::static_pointer_cast<TupleIndexingExpression>(woBlock).get(),
        prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    resolveStructExpression(std::static_pointer_cast<StructExpression>(woBlock),
                            prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    resolveCallExpression(std::static_pointer_cast<CallExpression>(woBlock),
                          prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    resolveMethodCallExpression(
        std::static_pointer_cast<MethodCallExpression>(woBlock).get(), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    resolveFieldExpression(
        std::static_pointer_cast<FieldExpression>(woBlock).get(), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    resolveClosureExpression(
        std::static_pointer_cast<ClosureExpression>(woBlock), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    resolveRangeExpression(std::static_pointer_cast<RangeExpression>(woBlock),
                           prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    resolveReturnExpression(std::static_pointer_cast<ReturnExpression>(woBlock),
                            prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    // FIXME macro expansion
    break;
  }
  }
}

void Resolver::resolveReturnExpression(
    std::shared_ptr<ast::ReturnExpression> returnExpr,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (returnExpr->hasTailExpression())
    resolveExpression(returnExpr->getExpression(), prefix, canonicalPrefix);
}

void Resolver::resolveOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> op,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    resolveBorrowExpression(std::static_pointer_cast<BorrowExpression>(op),
                            prefix, canonicalPrefix);
    break;
  }
  case OperatorExpressionKind::DereferenceExpression: {
    resolveDereferenceExpression(
        std::static_pointer_cast<DereferenceExpression>(op), prefix,
        canonicalPrefix);
    break;
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::NegationExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    resolveArithmeticOrLogicalExpression(
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op), prefix,
        canonicalPrefix);
    break;
  }
  case OperatorExpressionKind::ComparisonExpression: {
    resolveComparisonExpression(
        std::static_pointer_cast<ComparisonExpression>(op), prefix,
        canonicalPrefix);
    break;
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::TypeCastExpression: {
    resolveTypeCastExpression(std::static_pointer_cast<TypeCastExpression>(op),
                              prefix, canonicalPrefix);
    break;
  }
  case OperatorExpressionKind::AssignmentExpression: {
    std::shared_ptr<AssignmentExpression> assign =
        std::static_pointer_cast<AssignmentExpression>(op);
    resolveExpression(assign->getLHS(), prefix, canonicalPrefix);
    resolveExpression(assign->getRHS(), prefix, canonicalPrefix);
    verifyAssignee(assign->getLHS().get());
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    std::shared_ptr<ComparisonExpression> assign =
        std::static_pointer_cast<ComparisonExpression>(op);
    resolveExpression(assign->getLHS(), prefix, canonicalPrefix);
    resolveExpression(assign->getRHS(), prefix, canonicalPrefix);
    verifyAssignee(assign->getLHS().get());
    break;
  }
  }
}

void Resolver::resolvePathExpression(
    std::shared_ptr<ast::PathExpression> path, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    /*auto result =*/ resolvePathInExpression(
        std::static_pointer_cast<PathInExpression>(path), prefix,
        canonicalPrefix);
    //assert(result.has_value());
    break;
  }
  case PathExpressionKind::QualifiedPathInExpression: {
    resolveQualifiedPathInExpression(
        std::static_pointer_cast<QualifiedPathInExpression>(path), prefix,
        canonicalPrefix);
    break;
  }
  }
}

std::optional<NodeId>
Resolver::resolvePathInExpression(std::shared_ptr<ast::PathInExpression> path,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix) {
  NodeId resolvedNodeId = UNKNOWN_NODEID;
  NodeId moduleScopeId = peekCurrentModuleScope();
  NodeId previousResolvedNodeId = moduleScopeId;

  //  llvm::errs() << "resolve PathInExpression"
  //             << "\n";

  std::vector<PathExprSegment> segments = path->getSegments();

  for (size_t i = 0; i < segments.size(); ++i) {
    PathExprSegment &seg = segments[i];
    PathIdentSegment ident = seg.getIdent();

    if (i > 0 && ident.getKind() == PathIdentSegmentKind::self) {
      // report error
      llvm::errs() << path->getLocation().toString()
                   << llvm::formatv("failed to resolve {0} in paths can only "
                                    "be used in start position",
                                    ident.getIdentifier().toString())
                   << "\n";
      return std::nullopt;
    }

    NodeId crateScopeId = peekCrateModuleScope();
    if (ident.getKind() == PathIdentSegmentKind::crate) {
      moduleScopeId = crateScopeId;
      previousResolvedNodeId = moduleScopeId;
      insertResolvedName(seg.getNodeId(), moduleScopeId);
      continue;
    } else if (ident.getKind() == PathIdentSegmentKind::super) {
      if (moduleScopeId == crateScopeId) {
        llvm::errs() << " cannot super at the crate scope" << "\n";
        // report error
        return std::nullopt;
      }
      moduleScopeId = peekParentModuleScope();
      previousResolvedNodeId = moduleScopeId;
      insertResolvedName(seg.getNodeId(), moduleScopeId);
      continue;
    }

    if (seg.hasGenerics())
      resolveGenericArgs(seg.getGenerics(), prefix, canonicalPrefix);

    if (i == 0) {
      // name first
      // NodeId resolvedNode = UNKNOWN_NODEID;
      assert(ident.toString().size() != 0);
      CanonicalPath path = CanonicalPath::newSegment(
          seg.getNodeId(), lexer::Identifier(ident.toString()));
      if (auto node = getNameScope().lookup(path)) {
        // resolvedNode = *node;
        resolvedNodeId = *node;
      } else if (auto node = getTypeScope().lookup(path)) {
        insertResolvedType(seg.getNodeId(), *node);
        resolvedNodeId = *node;
      } else if (ident.getKind() == PathIdentSegmentKind::self) {
        moduleScopeId = crateScopeId;
        previousResolvedNodeId = moduleScopeId;
        insertResolvedName(seg.getNodeId(), moduleScopeId);
        continue;
      } else {
        // ?
      }
    }

    if (resolvedNodeId == UNKNOWN_NODEID &&
        previousResolvedNodeId == moduleScopeId) {
      std::optional<CanonicalPath> resolvedChild = tyCtx->lookupModuleChild(
          moduleScopeId, CanonicalPath::newSegment(
                             ident.getNodeId(), Identifier(ident.toString())));
      if (resolvedChild) {
        NodeId resolvedNode = resolvedChild->getNodeId();
        if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNode)) {
          resolvedNodeId = resolvedNode;
          insertResolvedName(seg.getNodeId(), resolvedNode);
        } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNode)) {
          resolvedNodeId = resolvedNode;
          insertResolvedType(seg.getNodeId(), resolvedNode);
        } else {
          // report error
          llvm::errs() << seg.getLocation().toString()
                       << llvm::formatv("cannot find path {0} in this scope",
                                        ident.getIdentifier().toString())
                       << "\n";
          return std::nullopt;
        }
      }
    }

    if (resolvedNodeId != UNKNOWN_NODEID) {
      if (tyCtx->isModule(resolvedNodeId) || tyCtx->isCrate(resolvedNodeId)) {
        moduleScopeId = resolvedNodeId;
      }
    } else if (i == 0) {
      // report error
      llvm::errs() << seg.getLocation().toString()
                   << llvm::formatv("cannot find path {0} in this scope",
                                    ident.getIdentifier().toString())
                   << "\n";
      return std::nullopt;
    }
  }

  // post for loop

  NodeId resolvedNode = resolvedNodeId;
  if (resolvedNodeId != UNKNOWN_NODEID) {
    // name first
    if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedName(path->getNodeId(), resolvedNodeId);
    } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedType(path->getNodeId(), resolvedNodeId);
    } else {
      // unreachable
    }
    return resolvedNode;
  }

  return std::nullopt;
}

void Resolver::resolveQualifiedPathInExpression(
    std::shared_ptr<ast::QualifiedPathInExpression> qual,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  QualifiedPathType root = qual->getType();
  resolveType(root.getType(), prefix, canonicalPrefix);
  if (root.hasPath())
    resolveType(root.getPath(), prefix, canonicalPrefix);

  for (PathExprSegment &segment : qual->getSegment())
    if (segment.hasGenerics())
      resolveGenericArgs(segment.getGenerics(), prefix, canonicalPrefix);
}

void Resolver::resolveBlockExpression(
    std::shared_ptr<ast::BlockExpression> block,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  NodeId scopeNodeId = block->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getNameScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  const Statements &stmts = block->getExpressions();

  for (auto &stmt : stmts.getStmts()) {
    resolveStatement(stmt, prefix, canonicalPrefix,
                     CanonicalPath::createEmpty());
  }

  if (stmts.hasTrailing())
    resolveExpression(stmts.getTrailing(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

void Resolver::verifyAssignee(ast::Expression *assignee) {
  switch (assignee->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    verifyAssignee(static_cast<ast::ExpressionWithBlock *>(assignee));
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    verifyAssignee(static_cast<ast::ExpressionWithoutBlock *>(assignee));
    break;
  }
  }
}

void Resolver::resolveArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(arith->getLHS(), prefix, canonicalPrefix);
  resolveExpression(arith->getRHS(), prefix, canonicalPrefix);
}

void Resolver::resolveIfExpression(std::shared_ptr<ast::IfExpression> ifExpr,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(ifExpr->getCondition(), prefix, canonicalPrefix);
  resolveExpression(ifExpr->getBlock(), prefix, canonicalPrefix);

  if (ifExpr->hasTrailing())
    resolveExpression(ifExpr->getTrailing(), prefix, canonicalPrefix);
}

void Resolver::resolveComparisonExpression(
    std::shared_ptr<ast::ComparisonExpression> cmp,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(cmp->getLHS(), prefix, canonicalPrefix);
  resolveExpression(cmp->getRHS(), prefix, canonicalPrefix);
}

void Resolver::resolveDereferenceExpression(
    std::shared_ptr<ast::DereferenceExpression> deref,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(deref->getRHS(), prefix, canonicalPrefix);
}

void Resolver::resolveBorrowExpression(
    std::shared_ptr<ast::BorrowExpression> borrow,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(borrow->getExpression(), prefix, canonicalPrefix);
}

void Resolver::resolveArrayExpression(
    std::shared_ptr<ast::ArrayExpression> arr, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (arr->hasArrayElements()) {
    ArrayElements el = arr->getArrayElements();
    switch (el.getKind()) {
    case ArrayElementsKind::List: {
      for (auto expr : el.getElements())
        resolveExpression(expr, prefix, canonicalPrefix);
      break;
    }
    case ArrayElementsKind::Repeated: {
      resolveExpression(el.getCount(), prefix, canonicalPrefix);
      resolveExpression(el.getValue(), prefix, canonicalPrefix);
      break;
    }
    }
  }
}

void Resolver::resolveIndexExpression(
    std::shared_ptr<ast::IndexExpression> indx,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(indx->getArray(), prefix, canonicalPrefix);
  resolveExpression(indx->getIndex(), prefix, canonicalPrefix);
}

void Resolver::resolveCallExpression(
    std::shared_ptr<ast::CallExpression> call, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(call->getFunction(), prefix, canonicalPrefix);

  if (call->hasParameters()) {
    CallParams params = call->getParameters();
    std::vector<std::shared_ptr<Expression>> pars = params.getParams();
    for (std::shared_ptr<Expression> p : pars)
      resolveExpression(p, prefix, canonicalPrefix);
  }
}

void Resolver::resolveMethodCallExpression(
    ast::MethodCallExpression *method, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(method->getReceiver(), prefix, canonicalPrefix);

  if (method->getMethod().hasGenerics()) {
    GenericArgs ga = method->getMethod().getGenerics();
    resolveGenericArgs(ga, prefix, canonicalPrefix);
  }

  CallParams pa = method->getParams();
  for (auto &param : pa.getParams())
    resolveExpression(param, prefix, canonicalPrefix);
}

void Resolver::resolveStructExpression(
    std::shared_ptr<StructExpression> str, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (str->getKind()) {
  case StructExpressionKind::StructExprStruct: {
    StructExprStruct *expr = static_cast<StructExprStruct *>(str.get());
    resolveExpression(expr->getName(), prefix, canonicalPrefix);

    if (expr->hasStructBase()) {
      StructBase base = expr->getStructBase();
      resolveExpression(base.getPath(), prefix, canonicalPrefix);
    }
    if (expr->hasStructExprFields())
      resolveStructExprFields(expr->getStructExprFields(), prefix,
                              canonicalPrefix);
    break;
  }
  case StructExpressionKind::StructExprTuple: {
    assert(false);
  }
  case StructExpressionKind::StructExprUnit: {
    StructExprUnit *expr = static_cast<StructExprUnit *>(str.get());
    resolveExpression(expr->getPath(), prefix, canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveStructExprFields(
    const StructExprFields &fields, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (fields.hasBase()) {
    StructBase base = fields.getBase();
    resolveExpression(base.getPath(), prefix, canonicalPrefix);
  }

  for (const StructExprField &field : fields.getFields()) {
    // FIXME: experiment
    if (field.hasIdentifier()) {
      CanonicalPath segment =
          CanonicalPath::newSegment(field.getNodeId(), field.getIdentifier());
      getNameScope().insert(segment, field.getNodeId(), field.getLocation(),
                            RibKind::Type);
    }
    if (field.hasExpression())
      resolveExpression(field.getExpression(), prefix, canonicalPrefix);
  }
}

void Resolver::resolveTupleExpression(
    ast::TupleExpression *tuple, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (tuple->isUnit())
    return;

  TupleElements elements = tuple->getElements();

  for (auto &element : elements.getElements())
    resolveExpression(element, prefix, canonicalPrefix);
}

void Resolver::resolveUnsafeBlockExpression(
    std::shared_ptr<ast::UnsafeBlockExpression> unsafe,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveBlockExpression(
      std::static_pointer_cast<BlockExpression>(unsafe->getBlock()), prefix,
      canonicalPrefix);
}

void Resolver::resolveTypeCastExpression(
    std::shared_ptr<TypeCastExpression> cast, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(cast->getLeft(), prefix, canonicalPrefix);
  resolveType(cast->getRight(), prefix, canonicalPrefix);
}

void Resolver::resolveRangeExpression(
    std::shared_ptr<ast::RangeExpression> range,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (range->getKind()) {
  case RangeExpressionKind::RangeExpr: {
    resolveExpression(range->getLeft(), prefix, canonicalPrefix);
    resolveExpression(range->getRight(), prefix, canonicalPrefix);
    break;
  }
  case RangeExpressionKind::RangeFromExpr: {
    resolveExpression(range->getLeft(), prefix, canonicalPrefix);
    break;
  }
  case RangeExpressionKind::RangeToExpr: {
    resolveExpression(range->getRight(), prefix, canonicalPrefix);
    break;
  }
  case RangeExpressionKind::RangeFullExpr: {
    break;
  }
  case RangeExpressionKind::RangeInclusiveExpr: {
    resolveExpression(range->getLeft(), prefix, canonicalPrefix);
    resolveExpression(range->getRight(), prefix, canonicalPrefix);
    break;
  }
  case RangeExpressionKind::RangeToInclusiveExpr: {
    resolveExpression(range->getRight(), prefix, canonicalPrefix);
    break;
  }
  }
}

void Resolver::verifyAssignee(ast::ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    // ok
    // FIXME
    PathExpression *path = static_cast<PathExpression *>(woBlock);
    switch (path->getPathExpressionKind()) {
    case PathExpressionKind::PathInExpression: {
      PathInExpression *pathIn = static_cast<PathInExpression *>(path);
      if (pathIn->isSingleSegment()) {
        PathExprSegment seg = pathIn->getSegments()[0];
        PathIdentSegment ident = seg.getIdent();
        assert(ident.getKind() == PathIdentSegmentKind::Identifier);
        lexer::Identifier id = ident.getIdentifier();
        std::optional<NodeId> nodeId = getNameScope().lookup(
            CanonicalPath::newSegment(path->getNodeId(), id));
        assert(nodeId.has_value());
        break;
      } else {
        break;
      }
      assert(false);
    }
    case PathExpressionKind::QualifiedPathInExpression: {
      assert(false);
    }
    }
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::verifyAssignee(ast::ExpressionWithBlock *withBlock) {
  assert(false);
}

void Resolver::resolveFieldExpression(
    ast::FieldExpression *field, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(field->getField(), prefix, canonicalPrefix);
}

void Resolver::resolveTupleIndexingExpression(
    ast::TupleIndexingExpression *tuple, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(tuple->getTuple(), prefix, canonicalPrefix);
}

void Resolver::resolveIfLetExpression(
    ast::IfLetExpression *ifLet, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  Scrutinee scrut = ifLet->getScrutinee();
  resolveExpression(scrut.getExpression(), prefix, canonicalPrefix);

  NodeId scopeNodeId = ifLet->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

  for (auto &path : ifLet->getPatterns()->getPatterns()) {
    PatternDeclaration pat = {
        path, RibKind::Variable, bindings, this, prefix,
        canonicalPrefix};
    pat.resolve();
    //    resolvePatternDeclarationWithBindings(path, RibKind::Variable,
    //    bindings,
    //                                          prefix, canonicalPrefix);
  }

  resolveExpression(ifLet->getBlock(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
