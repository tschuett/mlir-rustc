#pragma once

#include "Basic/Ids.h"

#include <cstdint>
#include <map>
#include <optional>

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
} // namespace rust_compiler::ast

namespace rust_compiler::constant_evaluation {

class ConstantEvaluation {
  const ast::Crate *crate;

  enum class OwnerKind { Expression, Statement, Item };
  class Owner {
    ast::Expression *expr;
    ast::Statement *stmt;
    ast::Item *item;
    OwnerKind kind;

  public:
    OwnerKind getKind() const { return kind; }

    ast::Item *getItem() const { return item; }

    static Owner expression(ast::Expression *expression) {
      Owner owner;
      owner.kind = OwnerKind::Expression;
      owner.expr = expression;
      return owner;
    }

    static Owner statement(ast::Statement *stmt) {
      Owner owner;
      owner.kind = OwnerKind::Statement;
      owner.stmt = stmt;
      return owner;
    }

    static Owner Item(ast::Item *item) {
      Owner owner;
      owner.kind = OwnerKind::Item;
      owner.item = item;
      return owner;
    }
  };

  std::map<basic::NodeId, Owner> owners;

public:
  ConstantEvaluation(const ast::Crate *crate) : crate(crate){};

  uint64_t foldAsUsize(const ast::Expression *);

private:
  uint64_t foldAsUsize(const ast::ExpressionWithBlock *);
  uint64_t foldAsUsize(const ast::ExpressionWithoutBlock *);
  uint64_t foldAsUsize(const ast::LiteralExpression *);
  uint64_t foldAsUsize(const ast::PathExpression *);
  uint64_t foldAsUsize(const ast::Item *);
  uint64_t foldAsUsize(const ast::VisItem *);
  uint64_t foldAsUsize(const ast::ConstantItem *);

  std::optional<Owner> getOwner(basic::NodeId id);
  std::optional<Owner> getOwner(basic::NodeId id, const ast::Crate *);
};

} // namespace rust_compiler::constant_evaluation
