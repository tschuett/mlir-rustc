#pragma once

#include "AST/BlockExpression.h"
#include "AST/Crate.h"
#include "AST/Decls.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/Statements.h"
#include "AST/Types/Types.h"
#include "Sema/Common.h"
#include "Sema/Mappings.h"
#include "Sema/TypeChecking.h"

#include <map>
#include <memory>

namespace rust_compiler::ast {
class Function;
class VisItem;
class Item;
class OuterAttributes;
class Expression;
class MacroInvocationSemi;
} // namespace rust_compiler::ast

namespace rust_compiler::sema {

class Sema {
public:
  void analyze(std::shared_ptr<ast::Crate> &ast);

private:
  void walkItem(std::shared_ptr<ast::Item> item);
  void walkVisItem(std::shared_ptr<ast::VisItem> item);
  void walkOuterAttributes(std::shared_ptr<ast::OuterAttributes>);

  void analyzeFunction(std::shared_ptr<ast::Function> fun);
  void analyzeBlockExpression(std::shared_ptr<ast::BlockExpression> block);
  void analyzeStatements(std::shared_ptr<ast::Statements> stmts);
  void analyzeLetStatement(std::shared_ptr<ast::LetStatement> let);
  void
  analyzeExpressionStatement(std::shared_ptr<ast::ExpressionStatement> expr);
  void
  analyzeMacroInvocationSemi(std::shared_ptr<ast::MacroInvocationSemi> macro);

  void analyzeItemDeclaration(std::shared_ptr<ast::Node> item);

  TypeChecking typechecking = {this};
  Mappings mappings = {this};
};

void analyzeSemantics(std::shared_ptr<ast::Crate> &ast);

} // namespace rust_compiler::sema
