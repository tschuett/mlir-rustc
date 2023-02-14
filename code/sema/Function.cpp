#include "AST/Function.h"

#include "AST/MacroInvocation.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::sema;

namespace rust_compiler::ast {
class ReturnExpression;
}

class BlockExpressionVisitor {
  Sema *sema;
  BlockExpression *block;

public:
  BlockExpressionVisitor(Sema *sema, BlockExpression *block)
      : sema(sema), block(block) {}

  void visit(Statement *);
  void visit(ExpressionWithoutBlock *);
  void visit(ExpressionWithBlock *);
  void visit(Item *);
  void visit(LetStatement *);
  void visit(ExpressionStatement *);
  void visit(MacroInvocation *);
  void visit(ReturnExpression *);

  void walk() {}
};

namespace rust_compiler::sema {

void Sema::analyzeFunction(std::shared_ptr<ast::Function> fun) {

}

} // namespace rust_compiler::sema
