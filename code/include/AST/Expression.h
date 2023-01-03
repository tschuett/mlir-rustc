#pragma once

#include "AST/AST.h"
#include "Location.h"

// #include <mlir/IR/Location.h>

namespace rust_compiler::ast {

enum class ExpressionKind { ExpressionWithBlock, ExpressionWithoutBlock };

class Expression : public Node {
public:
  Expression(rust_compiler::Location loc, ExpressionKind expressionKind)
      : loc(loc), expressionKind(expressionKind) {}

  ExpressionKind getExpressionKind() const { return expressionKind; }

  rust_compiler::Location getLocation() const { return loc; }

protected:
  rust_compiler::Location loc;
  ExpressionKind expressionKind;
};

enum class ExpressionWithoutBlockKind {
  LiteralExpression,
  PathExpression,
  OperationExpression,
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
  ExpressionWithBlockKind getKind() const { return kind; }

protected:
  ExpressionWithBlockKind kind;
};

} // namespace rust_compiler::ast
