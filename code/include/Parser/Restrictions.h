#pragma once

namespace rust_compiler::parser {

/* Restrictions on parsing used to signal that certain ambiguous grammar
 * features should be parsed in a certain way. */
struct ParseRestrictions {
  bool can_be_struct_expr = true;
  /* Whether the expression was entered from a unary expression - prevents stuff
   * like struct exprs being parsed from a dereference. */
  bool entered_from_unary = false;
  bool expr_can_be_null = false;
  bool expr_can_be_stmt = false;
  bool consume_semi = true;
};

struct Restrictions {
  bool canBeStructExpr = true;
  bool exprCanBeNull = false;
  /* Whether the expression was entered from a unary expression - prevents stuff
   * like struct exprs being parsed from a dereference. */
  bool enteredFromUnary = false;
};

} // namespace rust_commpiler::parser
