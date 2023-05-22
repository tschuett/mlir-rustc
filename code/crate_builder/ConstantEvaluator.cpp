#include "AST/Expression.h"
#include "AST/LiteralExpression.h"
#include "AST/PathExpression.h"
#include "AST/VisItem.h"
#include "CrateBuilder/CrateBuilder.h"

#include <cstdint>
#include <cstdlib>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

uint64_t CrateBuilder::foldAsUsizeExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    return foldAsUsizeWithBlock(static_cast<ExpressionWithBlock *>(expr));
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    return foldAsUsizeWithoutBlock(static_cast<ExpressionWithoutBlock *>(expr));
  }
  }
}

uint64_t
CrateBuilder::foldAsUsizeWithoutBlock(ast::ExpressionWithoutBlock *withOut) {
  switch (withOut->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    return foldAsUsizeLiteralExpression(
        static_cast<LiteralExpression *>(withOut));
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    return foldAsUsizePathExpression(static_cast<PathExpression *>(withOut));
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
  }
  }
}

uint64_t CrateBuilder::foldAsUsizeWithBlock(ast::ExpressionWithBlock *) {
  assert(false);
}

uint64_t CrateBuilder::foldAsUsizePathExpression(ast::PathExpression *path) {
  std::optional<basic::NodeId> resolvedPath =
      tyCtx->lookupName(path->getNodeId());

  if (resolvedPath) {
    std::optional<Owner> owner = getOwner(*resolvedPath);
    assert(owner.has_value());

    switch ((*owner).getKind()) {
    case OwnerKind::Expression: {
      assert(false);
    }
    case OwnerKind::Statement: {
      assert(false);
    }
    case OwnerKind::Item: {
      return foldAsUsizeItem((*owner).getItem());
      assert(false);
    }
    }
    assert(false);
    // uint64_t owned = foldAsUsizeNode(*owner);
    // return owned;
  }

  assert(false);
}

uint64_t CrateBuilder::foldAsUsizeItem(ast::Item *item) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    return foldAsUsizeVisItem(static_cast<ast::VisItem *>(item));
  }
  case ItemKind::MacroItem: {
    assert(false);
  }
  }
}

uint64_t CrateBuilder::foldAsUsizeVisItem(ast::VisItem *vis) {
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
    assert(false);
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
    return foldAsUsizeConstantItem(static_cast<ast::ConstantItem *>(vis));
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

uint64_t CrateBuilder::foldAsUsizeConstantItem(ast::ConstantItem *con) {

  if (con->hasInit()) {
    uint64_t init = foldAsUsizeExpression(con->getInit().get());
    return init;
  }
  assert(false);
}

uint64_t
CrateBuilder::foldAsUsizeLiteralExpression(ast::LiteralExpression *lit) {
  if (lit->getLiteralKind() != LiteralExpressionKind::IntegerLiteral) {
    llvm::errs() << "literal is not integer"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  llvm::APInt result;

  std::string storage = lit->getValue();
  llvm::StringRef(storage).getAsInteger(10, result);

  unsigned bitWidth = result.getBitWidth();

  if (bitWidth <= 64)
    return result.getLimitedValue();

  llvm::errs() << "failed to fold literal"
               << "\n";

  exit(EXIT_FAILURE);
}

} // namespace rust_compiler::crate_builder
