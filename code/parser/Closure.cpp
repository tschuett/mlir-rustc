#include "AST/ClosureExpression.h"
#include "AST/ClosureParam.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseClosureExpression() {
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
    llvm::Expected<ast::ClosureParameters> parameters =
        parseClosureParameters();
    if (auto e = parameters.takeError()) {
      llvm::errs() << "failed to parse closure parameters in closure: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    clos.setParameters(*parameters);
  } else {
    // error
  }

  if (check(TokenKind::RArrow)) {
    assert(eat(TokenKind::RArrow));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> types =
        parseTypeNoBounds();
    if (auto e = types.takeError()) {
      llvm::errs() << "failed to parse type in closure: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    clos.setType(*types);
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (auto e = block.takeError()) {
      llvm::errs() << "failed to parse block expression in closure: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    clos.setBlock(*block);
    return std::make_shared<ClosureExpression>(clos);
  } else {
    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression in closure: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    clos.setExpr(*expr);
    return std::make_shared<ClosureExpression>(clos);
  }

  // error
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse closure expression ");
}

llvm::Expected<ast::ClosureParameters> Parser::parseClosureParameters() {
  Location loc = getLocation();

  ClosureParameters params = {loc};

  llvm::Expected<ast::ClosureParam> first = parseClosureParam();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse closure param in closure parameters: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  params.addParam(*first);

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse closure parameters: eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::Or, 1)) {
      // done
      return params;
    } else if (check(TokenKind::Or)) {
      // done
      return params;
    } else if (check(TokenKind::Or)) {
      llvm::Expected<ast::ClosureParam> cp = parseClosureParam();
      if (auto e = cp.takeError()) {
        llvm::errs() << "failed to parse closure param in closure parameters: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      params.addParam(*cp);
    }
  }
  // error
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse closure parameters");
}

llvm::Expected<ast::ClosureParam> Parser::parseClosureParam() {
  Location loc = getLocation();

  ClosureParam param = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in closure param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setOuterAttributes(*outer);
  }

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt in closure param: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.setPattern(*pattern);

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));

    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type expression in closure param: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setType(*type);
  }

  return param;
}

} // namespace rust_compiler::parser
