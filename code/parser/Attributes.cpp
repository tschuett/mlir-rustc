#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <sstream>
#include <vector>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::OuterAttribute> Parser::parseOuterAttribute() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  Location loc = getLocation();

  OuterAttribute outer = {loc};

  if (!check(TokenKind::Hash))
    return StringResult<ast::OuterAttribute>("failed to parse outer attribute");
  assert(eat(TokenKind::Hash));

  if (!check(TokenKind::SquareOpen))
    return StringResult<ast::OuterAttribute>("failed to parse outer attribute");
  assert(eat(TokenKind::SquareOpen));

  StringResult<Attr> attr = parseAttr();
  if (!attr) {
    llvm::errs() << "failed to parse attr  in outer attribute: "
                 << attr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  outer.setAttr(attr.getValue());

  if (!check(TokenKind::SquareClose))
    return StringResult<ast::OuterAttribute>("failed to parse outer attribute");
  assert(eat(TokenKind::SquareClose));

  return StringResult<ast::OuterAttribute>(outer);
}

StringResult<ast::InnerAttribute> Parser::parseInnerAttribute() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  Location loc = getLocation();

  InnerAttribute inner = {loc};

  if (!check(TokenKind::Hash))
    return StringResult<ast::InnerAttribute>(
                             "failed to parse inner attribute");
  assert(eat(TokenKind::Hash));

  if (!check(TokenKind::Not))
    return StringResult<ast::InnerAttribute>(
                             "failed to parse inner attribute");
  assert(eat(TokenKind::Not));

  if (!check(TokenKind::SquareOpen))
    return StringResult<ast::InnerAttribute>(
                             "failed to parse inner attribute");
  assert(eat(TokenKind::SquareOpen));

  StringResult<Attr> attr = parseAttr();
  if (!attr) {
    llvm::errs() << "failed to parse attr  in inner attribute: "
                 << attr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  inner.setAttr(attr.getValue());

  if (!check(TokenKind::SquareClose))
    return StringResult<ast::InnerAttribute>(
                             "failed to parse inner attribute");
  assert(eat(TokenKind::SquareClose));

  return StringResult<ast::InnerAttribute>(inner);
}

StringResult<ast::Attr> Parser::parseAttr() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  Attr attr = {loc};

  StringResult<ast::SimplePath> path = parseSimplePath();
  if (!path) {
    llvm::errs() << "failed to parse simple path  in attr: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  attr.setSimplePath(path.getValue());

  if (check(TokenKind::SquareClose)) {
  } else {
    StringResult<ast::AttrInput> attrInput = parseAttrInput();
    if (!attrInput) {
      llvm::errs() << "failed to parse attr input in attr: "
                   << attrInput.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    attr.setAttrInput(attrInput.getValue());
  }
  return StringResult<ast::Attr>(attr);
}

StringResult<ast::AttrInput> Parser::parseAttrInput() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  AttrInput input = {loc};

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    Restrictions restrictions;
    Result<std::shared_ptr<ast::Expression>, std::string> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in AttrInput: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    input.setExpression(expr.getValue());
    return StringResult<ast::AttrInput>(input);
  }

  StringResult<std::shared_ptr<ast::DelimTokenTree>> tokenTree =
      parseDelimTokenTree();
  if (!tokenTree) {
    llvm::errs() << "failed to parse delim token tree in attr input: "
                 << tokenTree.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  input.setTokenTree(tokenTree.getValue());
  return StringResult<ast::AttrInput>(input);
}

StringResult<std::vector<ast::OuterAttribute>> Parser::parseOuterAttributes() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  std::vector<OuterAttribute> outer;

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<std::vector<ast::OuterAttribute>>(
          "failed to parse outer attributes");
    } else if (checkOuterAttribute()) {
      StringResult<ast::OuterAttribute> outerAttr = parseOuterAttribute();
      if (!outerAttr) {
        llvm::errs() << "failed to parse outer attribute in outer attributes: "
                     << outerAttr.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      outer.push_back(outerAttr.getValue());
    } else {
      return StringResult<std::vector<ast::OuterAttribute>>(outer);
    }
  }
}

StringResult<std::vector<ast::InnerAttribute>> Parser::parseInnerAttributes() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  std::vector<InnerAttribute> inner;

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<std::vector<ast::InnerAttribute>>(
          "failed to parse inner attributes");
    } else if (checkInnerAttribute()) {
      StringResult<ast::InnerAttribute> innerAttr = parseInnerAttribute();
      if (!innerAttr) {
        llvm::errs() << "failed to parse inner attribute in inner attributes: "
                     << innerAttr.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      inner.push_back(innerAttr.getValue());
    } else {
      return StringResult<std::vector<ast::InnerAttribute>>(inner);
    }
  }
}

} // namespace rust_compiler::parser
