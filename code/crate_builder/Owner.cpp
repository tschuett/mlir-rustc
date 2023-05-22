#include "AST/Expression.h"
#include "AST/ItemDeclaration.h"
#include "AST/Statement.h"
#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"

#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

std::optional<CrateBuilder::Owner> CrateBuilder::getOwner(basic::NodeId id) {
  std::optional<Owner> owner = getOwnerCrate(id);

  if (owner) {
    owners[id] = *owner;
    return owner;
  }

  llvm::errs() << "owner failed"
               << "\n";


  return std::nullopt;
  // FIXME caching
}

std::optional<CrateBuilder::Owner> CrateBuilder::getOwnerCrate(basic::NodeId id) {
  assert(id != crate->getNodeId());

  for (auto &item : crate->getItems()) {
    std::optional<Owner> found = getOwnerItem(id, item.get());
    if (found)
      return *found;
  }

    llvm::errs() << "crate failed"
               << "\n";

  
  return std::nullopt;
}

std::optional<CrateBuilder::Owner> CrateBuilder::getOwnerItem(basic::NodeId id,
                                                 ast::Item *item) {
  if (item->getNodeId() == id)
    return Owner::Item(item);

  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    std::optional<Owner> visItem =
        getOwnerVisItem(id, static_cast<VisItem *>(item));
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

std::optional<CrateBuilder::Owner> CrateBuilder::getOwnerVisItem(basic::NodeId id,
                                                    VisItem *vis) {

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
    return getOwnerFunction(id, static_cast<Function *>(vis));
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

std::optional<CrateBuilder::Owner> CrateBuilder::getOwnerFunction(basic::NodeId id,
                                                          ast::Function *fun) {
  if (fun->getNodeId() == id)
    return Owner::Item(fun);

  if (fun->hasBody()) {
    std::optional<Owner> node =
        getOwnerExpression(id, fun->getBody().get());
    if (node) {
      return *node;
    }
  }

  // FIXME extend

  llvm::errs() << "failed function"
               << "\n";

  return std::nullopt;
}

std::optional<CrateBuilder::Owner>
CrateBuilder::getOwnerExpression(basic::NodeId id, ast::Expression *expr) {
  if (expr->getNodeId() == id)
    return Owner::expression(expr);

  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    std::optional<Owner> n = getOwnerExpressionWithBlock(
        id, static_cast<ExpressionWithBlock *>(expr));
    if (n)
      return *n;
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    std::optional<Owner> n = getOwnerExpressionWithoutBlock(
        id, static_cast<ExpressionWithoutBlock *>(expr));
    if (n)
      return *n;
    break;
  }
  }

  llvm::errs() << "failed expression"
               << "\n";
  return std::nullopt;
}

std::optional<CrateBuilder::Owner>
CrateBuilder::getOwnerExpressionWithBlock(basic::NodeId id,
                                          ast::ExpressionWithBlock *withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    std::optional<Owner> block =
        getOwnerBlockExpression(id, static_cast<BlockExpression *>(withBlock));
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

std::optional<CrateBuilder::Owner> CrateBuilder::getOwnerExpressionWithoutBlock(
    basic::NodeId id, ast::ExpressionWithoutBlock *withOutBlock) {
  assert(false);
}

std::optional<CrateBuilder::Owner>
CrateBuilder::getOwnerBlockExpression(basic::NodeId id,
                                      ast::BlockExpression *block) {
  if (block->getNodeId() == id)
    return Owner::expression(block);

  Statements stmts = block->getExpressions();

  llvm::errs() << "number of stmts: " << stmts.getNrOfStatements() << "\n";

  for (auto &stmt : stmts.getStmts()) {
    llvm::errs() << "statement"
                 << "\n";
    std::optional<Owner> st = getOwnerStatement(id, stmt.get());
    if (st) {
      llvm::errs() << "statement: success"
                   << "\n";
      return *st;
    }
  }

  if (stmts.hasTrailing()) {
    std::optional<Owner> st =
        getOwnerExpression(id, stmts.getTrailing().get());
    if (st)
      return *st;
  }

  return std::nullopt;
}

std::optional<CrateBuilder::Owner>
CrateBuilder::getOwnerStatement(basic::NodeId id, ast::Statement *stmt) {
  if (stmt->getNodeId() == id)
    return Owner::statement(stmt);

  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    assert(false);
  }
  case StatementKind::ItemDeclaration: {
    std::optional<Owner> item =
        getOwnerItemDeclaration(id, static_cast<ItemDeclaration *>(stmt));
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

std::optional<CrateBuilder::Owner>
CrateBuilder::getOwnerItemDeclaration(basic::NodeId id,
                                      ast::ItemDeclaration *item) {
  if (item->getNodeId() == id)
    return Owner::statement(item);

  if (item->hasVisItem()) {
    std::optional<Owner> vis = getOwnerVisItem(id, item->getVisItem().get());
    if (vis) {
      return *vis;
    }
  } else if (item->hasMacroItem()) {
    assert(false);
  }

  return std::nullopt;
}

} // namespace rust_compiler::crate_builder
