#pragma once

#include "AST/BlockExpression.h"
#include "AST/ItemDeclaration.h"
#include "Basic/Ids.h"
#include "Session/Session.h"
#include "TyCtx/TyCtx.h"

#include <cstdint>
#include <map>
#include <optional>

/// https://doc.rust-lang.org/stable/reference/const_eval.html
namespace rust_compiler::ast {
class Expression;
class ExpressionWithBlock;
class ExpressionWithoutBlock;
class LiteralExpression;
class PathExpression;
class Crate;
class Statement;
class Item;
class VisItem;
class ConstantItem;
class ArithmeticOrLogicalExpression;
class OperatorExpression;
class ComparisonExpression;
} // namespace rust_compiler::ast

namespace rust_compiler::constant_evaluation {

enum class OwnerKind { Expression, Statement, Item };
class Owner {
  const ast::Expression *expr;
  const ast::Statement *stmt;
  const ast::Item *item;
  OwnerKind kind;

public:
  OwnerKind getKind() const { return kind; }

  const ast::Item *getItem() const { return item; }

  static Owner expression(const ast::Expression *expression) {
    Owner owner;
    owner.kind = OwnerKind::Expression;
    owner.expr = expression;
    return owner;
  }

  static Owner statement(const ast::Statement *stmt) {
    Owner owner;
    owner.kind = OwnerKind::Statement;
    owner.stmt = stmt;
    return owner;
  }

  static Owner Item(const ast::Item *item) {
    Owner owner;
    owner.kind = OwnerKind::Item;
    owner.item = item;
    return owner;
  }
};

class ConstantEvaluation {
  const ast::Crate *crate;

  std::map<basic::NodeId, Owner> owners;

  tyctx::TyCtx *tyCtx;

public:
  ConstantEvaluation(const ast::Crate *crate) : crate(crate) {
    tyCtx = rust_compiler::session::session->getTypeContext();
  };

  uint64_t foldAsUsize(const ast::Expression *);

private:
  uint64_t foldAsUsize(const ast::ExpressionWithBlock *);
  uint64_t foldAsUsize(const ast::ExpressionWithoutBlock *);
  uint64_t foldAsUsize(const ast::LiteralExpression *);
  uint64_t foldAsUsize(const ast::PathExpression *);
  uint64_t foldAsUsize(const ast::Item *);
  uint64_t foldAsUsize(const ast::VisItem *);
  uint64_t foldAsUsize(const ast::ConstantItem *);
  uint64_t foldAsUsize(const ast::OperatorExpression *);
  uint64_t foldAsUsize(const ast::ArithmeticOrLogicalExpression *);
  uint64_t foldAsUsize(const ast::ComparisonExpression *);

  std::optional<Owner> getOwner(basic::NodeId id);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Crate *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Item *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::VisItem *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Function *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Expression *);
  std::optional<Owner> getOwner(basic::NodeId id,
                                const ast::ExpressionWithBlock *);
  std::optional<Owner> getOwner(basic::NodeId id,
                                const ast::ExpressionWithoutBlock *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::BlockExpression *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Statement *);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::ItemDeclaration *);
};

} // namespace rust_compiler::constant_evaluation
