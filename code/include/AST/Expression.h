#pragma once

#include "AST/AST.h"
#include "Location.h"

// #include <mlir/IR/Location.h>

namespace rust_compiler::ast {

enum class ExpressionKind { ExpressionWithBlock, ExpressionWithoutBlock };

class Expression : public Node {
public:
  Expression(rust_compiler::Location loc, ExpressionKind expressionKind)
      : Node(loc), loc(loc), expressionKind(expressionKind) {}

  ExpressionKind getExpressionKind() const { return expressionKind; }

  rust_compiler::Location getLocation() const { return loc; }

  void setHasTrailingSemi() { hasTrailingSemi = true; }

protected:
  rust_compiler::Location loc;
  ExpressionKind expressionKind;
  bool hasTrailingSemi = false;
};

enum class ExpressionWithoutBlockKind {
  LiteralExpression,
  PathExpression,
  OperatorExpression,
  GroupedExpression,
  ArrayExpression,
  AwaitExpression,
  IndexExpression,
  TupleExpression,
  TupleIndexingExpression,
  StructExpression,
  CallExpression,
  MethodCallExpression,
  FieldExpression,
  ClosureExpression,
  AsyncBlockExpression,
  ContinueExpression,
  BreakExpression,
  RangeExpression,
  ReturnExpression,
  UnderScoreExpression,
  MacroInvocation
};

class ExpressionWithoutBlock : public Expression {
public:
  ExpressionWithoutBlock(Location loc, ExpressionWithoutBlockKind kind)
: Expression(loc, ExpressionKind::ExpressionWithBlock), kind(kind) {}

  ExpressionWithoutBlockKind getKind() const { return kind; }

protected:
  ExpressionWithoutBlockKind kind;
};

enum class ExpressionWithBlockKind {
  BlockExpression,
  UnsafeBlockExpression,
  LoopExpression,
  IfExpression,
  IfLetExpression,
  MatchExpression
};

class ExpressionWithBlock : public Expression {

public:
  ExpressionWithBlock(Location loc, ExpressionWithBlockKind kind)
      : Expression(loc, ExpressionKind::ExpressionWithBlock), kind(kind) {}
  ExpressionWithBlockKind getKind() const { return kind; }

protected:
  ExpressionWithBlockKind kind;
};

} // namespace rust_compiler::ast
