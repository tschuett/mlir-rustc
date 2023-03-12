#include "AST/Expression.h"

using namespace rust_compiler::ast;

namespace rust_compiler::ast {

std::string ExpressionWithBlockKind2String(ExpressionWithBlockKind kind) {
  switch (kind) {
  case ExpressionWithBlockKind::BlockExpression:
    return "block expression";
  case ExpressionWithBlockKind::UnsafeBlockExpression:
    return "unsafe block expression";
  case ExpressionWithBlockKind::LoopExpression:
    return "loop expression";
  case ExpressionWithBlockKind::IfExpression:
    return "if expression";
  case ExpressionWithBlockKind::IfLetExpression:
    return "if let expression";
  case ExpressionWithBlockKind::MatchExpression:
    return "matc expression";
  }
}

} // namespace rust_compiler::ast
