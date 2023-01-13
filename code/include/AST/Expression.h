#pragma once

#include "AST/AST.h"
#include "Location.h"
#include "AST/Types/Types.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::ast {

enum class ExpressionKind { ExpressionWithBlock, ExpressionWithoutBlock };

class Expression : public Node {
public:
  Expression(rust_compiler::Location loc, ExpressionKind expressionKind)
      : Node(loc), loc(loc), expressionKind(expressionKind) {}

  ExpressionKind getExpressionKind() const { return expressionKind; }

  rust_compiler::Location getLocation() const { return loc; }

  void setHasTrailingSemi() { hasTrailingSemi = true; }

  bool getHasTrailingSemi() { return hasTrailingSemi; }

  virtual std::shared_ptr<ast::types::Type> getType() = 0;

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

enum class ExpressionWithBlockKind : uint32_t {
  Unknown = 0,
  BlockExpression = 1,
  UnsafeBlockExpression = 2,
  LoopExpression = 3,
  IfExpression = 4,
  IfLetExpression = 5,
  MatchExpression = 6
};

class ExpressionWithBlock : public Expression {

public:
  ExpressionWithBlock() = delete;

  ExpressionWithBlock(Location loc, ExpressionWithBlockKind withKind)
      : Expression(loc, ExpressionKind::ExpressionWithBlock),
        withKind(withKind) {}
  ExpressionWithBlockKind getWithBlockKind() const { return withKind; }

private:
  ExpressionWithBlockKind withKind = ExpressionWithBlockKind::Unknown;
};

} // namespace rust_compiler::ast
