#pragma once

#include "AST/AST.h"
//#include "AST/Types/Types.h"
//#include "Location.h"

//#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::ast {

enum class ExpressionKind { ExpressionWithBlock, ExpressionWithoutBlock };

class Expression : public Node {
public:
  Expression(rust_compiler::Location loc, ExpressionKind expressionKind)
      : Node(loc), loc(loc), expressionKind(expressionKind) {}

  ExpressionKind getExpressionKind() const { return expressionKind; }

  void setHasTrailingSemi() { hasTrailingSemi = true; }

  bool getHasTrailingSemi() { return hasTrailingSemi; }

private:
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
  ExpressionWithoutBlock(Location loc, ExpressionWithoutBlockKind withoutKind)
      : Expression(loc, ExpressionKind::ExpressionWithoutBlock),
        withoutKind(withoutKind) {}

  ExpressionWithoutBlockKind getWithoutBlockKind() const { return withoutKind; }

private:
  ExpressionWithoutBlockKind withoutKind;
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
  ExpressionWithBlock() = delete;

  ExpressionWithBlock(Location loc, ExpressionWithBlockKind withKind)
      : Expression(loc, ExpressionKind::ExpressionWithBlock),
        withKind(withKind) {}
  ExpressionWithBlockKind getWithBlockKind() const { return withKind; }

private:
  ExpressionWithBlockKind withKind;
};

} // namespace rust_compiler::ast
