#pragma once

#include <map>
#include <mlir/IR/Block.h>
#include <span>

namespace rust_compiler::analysis {

class LoopNest;

enum class ScevExpressionKind { Addition, Constant };

class SCEVExpression {};
class SCEVAddExpression : public SCEVExpression {};
class SCEVConstantExpression : public SCEVExpression {};

class ScalarEvolution {
  std::map<mlir::Value *, SCEVExpression *> scevs;

public:
  void analyze(std::span<LoopNest>);

private:
  void analyzeLoopNest(LoopNest *);
};

} // namespace rust_compiler::analysis
