#include "AST/ClosureExpression.h"
#include "AST/ClosureParam.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "Parser/Restrictions.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Expression>>
Parser::parseClosureExpression(std::span<ast::OuterAttribute> outer) {
  Location loc = getLocation();

  ClosureExpression clos = {loc};

  if (checkKeyWord(KeyWordKind::KW_MOVE)) {
    clos.setMove();
    assert(eatKeyWord(KeyWordKind::KW_MOVE));
  }

  if (check(TokenKind::Or) && check(TokenKind::OrOr, 1)) {
    // no parameters
  } else if (check(TokenKind::Or) && !!check(TokenKind::OrOr)) {
    assert(check(TokenKind::Or));
    // parameters
    StringResult<ast::ClosureParameters> parameters = parseClosureParameters();
    if (!parameters) {
      llvm::errs()
          << "failed to parse closure parameters in closure expression: "
          << parameters.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    clos.setParameters(parameters.getValue());
  } else {
    // error
  }

  if (check(TokenKind::RArrow)) {
    assert(eat(TokenKind::RArrow));
    StringResult<std::shared_ptr<ast::types::TypeExpression>> types =
        parseTypeNoBounds();
    if (!types) {
      llvm::errs() << "failed to parse type no bounds in closure expression: "
                   << types.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    clos.setType(types.getValue());
    StringResult<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (!block) {
      llvm::errs()
          << "failed to parse type noblock expression in closure expression: "
          << block.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    clos.setBlock(block.getValue());
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<ClosureExpression>(clos));
  } else {
    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> expr = parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse  expression in closure expression: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    clos.setExpr(expr.getValue());
    return StringResult<std::shared_ptr<ast::Expression>>(
        std::make_shared<ClosureExpression>(clos));
  }

  // error
  return StringResult<std::shared_ptr<ast::Expression>>(
      "failed to parse closure expression ");
}

StringResult<ast::ClosureParameters> Parser::parseClosureParameters() {
  Location loc = getLocation();

  ClosureParameters params = {loc};

  StringResult<ast::ClosureParam> first = parseClosureParam();
  if (!first) {
    llvm::errs() << "failed to parse  closure param in closure parameters: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  params.addParam(first.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<ast::ClosureParameters>(
          "failed to parse closure parameters: eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::Or, 1)) {
      // done
      return StringResult<ast::ClosureParameters>(params);
    } else if (check(TokenKind::Or)) {
      // done
      return StringResult<ast::ClosureParameters>(params);
    } else if (check(TokenKind::Or)) {
      StringResult<ast::ClosureParam> cp = parseClosureParam();
      if (!cp) {
        llvm::errs() << "failed to parse  closure param in closure parameters: "
                     << cp.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      params.addParam(cp.getValue());
    }
  }
  // error
  return StringResult<ast::ClosureParameters>(
      "failed to parse closure parameters");
}

StringResult<ast::ClosureParam> Parser::parseClosureParam() {
  Location loc = getLocation();

  ClosureParam param = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in closure param: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> ot = outer.getValue();
    param.setOuterAttributes(ot);
  }

  StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in closure param: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.setPattern(pattern.getValue());

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));

    StringResult<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to parse type in closure param: "
                   << type.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    param.setType(type.getValue());
  }

  return StringResult<ast::ClosureParam>(param);
}

} // namespace rust_compiler::parser
