#include "AST/Crate.h"
#include "AST/VisItem.h"
#include "ConstantEvaluation/ConstantEvaluation.h"

using namespace rust_compiler::ast;

namespace rust_compiler::constant_evaluation {

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id) {
  std::optional<Owner> owner = getOwner(id, crate);

  if (owner) {
    owners[id] = *owner;
    return owner;
  }

  llvm::errs() << "owner failed"
               << "\n";

  return std::nullopt;
  // FIXME caching
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Crate *crate) {
  assert(id != crate->getNodeId());

  for (auto &item : crate->getItems()) {
    std::optional<Owner> found = getOwner(id, item.get());
    if (found)
      return *found;
  }

  llvm::errs() << "crate failed"
               << "\n";

  return std::nullopt;
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Item *item) {
  if (item->getNodeId() == id)
    return Owner::Item(item);

  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    std::optional<Owner> visItem =
        getOwner(id, static_cast<const VisItem *>(item));
    if (visItem)
      return *visItem;
    break;
  }
  case ItemKind::MacroItem: {
    assert(false);
  }
  }

  llvm::errs() << "item failed"
               << "\n";
  return std::nullopt;
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::VisItem *vis) {

  if (vis->getNodeId() == id)
    return Owner::Item(vis);

  switch (vis->getKind()) {
  case VisItemKind::Module: {
    assert(false);
  }
  case VisItemKind::ExternCrate: {
    assert(false);
  }
  case VisItemKind::UseDeclaration: {
    assert(false);
  }
  case VisItemKind::Function: {
    return getOwner(id, static_cast<const Function *>(vis));
  }
  case VisItemKind::TypeAlias: {
    assert(false);
  }
  case VisItemKind::Struct: {
    assert(false);
  }
  case VisItemKind::Enumeration: {
    assert(false);
  }
  case VisItemKind::Union: {
    assert(false);
  }
  case VisItemKind::ConstantItem: {
    assert(false);
  }
  case VisItemKind::StaticItem: {
    assert(false);
  }
  case VisItemKind::Trait: {
    assert(false);
  }
  case VisItemKind::Implementation: {
    assert(false);
  }
  case VisItemKind::ExternBlock: {
    assert(false);
  }
  }
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Function *fun) {
  if (fun->getNodeId() == id)
    return Owner::Item(fun);

  if (fun->hasBody()) {
    std::optional<Owner> node = getOwner(id, fun->getBody().get());
    if (node) {
      return *node;
    }
  }

  // FIXME extend

  llvm::errs() << "failed function"
               << "\n";

  return std::nullopt;
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Expression *expr) {
  if (expr->getNodeId() == id)
    return Owner::expression(expr);

  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    std::optional<Owner> n =
        getOwner(id, static_cast<const ExpressionWithBlock *>(expr));
    if (n)
      return *n;
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    std::optional<Owner> n =
        getOwner(id, static_cast<const ExpressionWithoutBlock *>(expr));
    if (n)
      return *n;
    break;
  }
  }

  llvm::errs() << "failed expression"
               << "\n";
  return std::nullopt;
}

std::optional<Owner>
ConstantEvaluation::getOwner(basic::NodeId id,
                             const ast::ExpressionWithBlock *withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    std::optional<Owner> block =
        getOwner(id, static_cast<const BlockExpression *>(withBlock));
    if (block) {
      return *block;
    }
    assert(false);
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    assert(false);
  }
  case ExpressionWithBlockKind::LoopExpression: {
    assert(false);
  }
  case ExpressionWithBlockKind::IfExpression: {
    assert(false);
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    assert(false);
  }
  case ExpressionWithBlockKind::MatchExpression: {
    assert(false);
  }
  }
}

std::optional<Owner>
ConstantEvaluation::getOwner(basic::NodeId id,
                             const ast::ExpressionWithoutBlock *) {
  assert(false);
}

std::optional<Owner>
ConstantEvaluation::getOwner(basic::NodeId id,
                             const ast::BlockExpression *block) {
  if (block->getNodeId() == id)
    return Owner::expression(block);

  Statements stmts = block->getExpressions();

  for (auto &stmt : stmts.getStmts()) {
    llvm::errs() << "statement"
                 << "\n";
    std::optional<Owner> st = getOwner(id, stmt.get());
    if (st) {
      return *st;
    }
  }

  if (stmts.hasTrailing()) {
    std::optional<Owner> st = getOwner(id, stmts.getTrailing().get());
    if (st)
      return *st;
  }

  return std::nullopt;
}

std::optional<Owner> ConstantEvaluation::getOwner(basic::NodeId id,
                                                  const ast::Statement *stmt) {
  if (stmt->getNodeId() == id)
    return Owner::statement(stmt);

  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    assert(false);
  }
  case StatementKind::ItemDeclaration: {
    std::optional<Owner> item =
        getOwner(id, static_cast<const ItemDeclaration *>(stmt));
    if (item) {
      return *item;
    }
    assert(false);
  }
  case StatementKind::LetStatement: {
    assert(false);
  }
  case StatementKind::ExpressionStatement: {
    assert(false);
  }
  case StatementKind::MacroInvocationSemi: {
    assert(false);
  }
  }
}

std::optional<Owner>
ConstantEvaluation::getOwner(basic::NodeId id, const ast::ItemDeclaration *item) {
  if (item->getNodeId() == id)
    return Owner::statement(item);

  if (item->hasVisItem()) {
    std::optional<Owner> vis = getOwner(id, item->getVisItem().get());
    if (vis) {
      return *vis;
    }
  } else if (item->hasMacroItem()) {
    assert(false);
  }

  return std::nullopt;
}

} // namespace rust_compiler::constant_evaluation
