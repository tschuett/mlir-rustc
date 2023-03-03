#include "Parser/Parser.h"

#include <optional>
#include <sstream>
#include <vector>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::OuterAttribute> Parser::parseOuterAttribute() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  Location loc = getLocation();

  OuterAttribute outer = {loc};

  if (!check(TokenKind::Hash))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse outer attribute");
  assert(eat(TokenKind::Hash));

  if (!check(TokenKind::SquareOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse outer attribute");
  assert(eat(TokenKind::SquareOpen));

  llvm::Expected<Attr> attr = parseAttr();

  if (auto e = attr.takeError()) {
    llvm::errs() << "failed to parse attr in outer attribute: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  outer.setAttr(*attr);

  if (!check(TokenKind::SquareClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse outer attribute");
  assert(eat(TokenKind::SquareClose));

  return outer;
}

llvm::Expected<ast::InnerAttribute> Parser::parseInnerAttribute() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  Location loc = getLocation();

  InnerAttribute inner = {loc};

  if (!check(TokenKind::Hash))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse inner attribute");
  assert(eat(TokenKind::Hash));

  if (!check(TokenKind::Not))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse inner attribute");
  assert(eat(TokenKind::Not));

  if (!check(TokenKind::SquareOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse inner attribute");
  assert(eat(TokenKind::SquareOpen));

  llvm::Expected<Attr> attr = parseAttr();

  if (auto e = attr.takeError()) {
    llvm::errs() << "failed to parse attr in inner attribute: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  inner.setAttr(*attr);

  if (!check(TokenKind::SquareClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse inner attribute");
  assert(eat(TokenKind::SquareClose));

  return inner;
}

llvm::Expected<ast::Attr> Parser::parseAttr() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  Attr attr = {loc};

  llvm::Expected<ast::SimplePath> path = parseSimplePath();
  if (auto e = path.takeError()) {
    llvm::errs() << "failed to parse simple path in Attr: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  attr.setSimplePath(*path);

  if (check(TokenKind::SquareClose)) {
  } else {
    llvm::Expected<ast::AttrInput> attrInput = parseAttrInput();
    if (auto e = attrInput.takeError()) {
      llvm::errs() << "failed to parse attr input in Attr: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    attr.setAttrInput(*attrInput);
  }
  return attr;
}

llvm::Expected<ast::AttrInput> Parser::parseAttrInput() {
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
    return input;
  }

  llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> tokenTree =
      parseDelimTokenTree();
  if (auto e = tokenTree.takeError()) {
    llvm::errs() << "failed to parse token tree in AttrInput: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  input.setTokenTree(*tokenTree);
  return input;
}

llvm::Expected<std::vector<ast::OuterAttribute>>
Parser::parseOuterAttributes() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  std::vector<OuterAttribute> outer;

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse outer attributes");
    } else if (checkOuterAttribute()) {
      llvm::Expected<ast::OuterAttribute> outerAttr = parseOuterAttribute();
      if (auto e = outerAttr.takeError()) {
        llvm::errs() << "failed to parse outer attribute in outer attributes: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      outer.push_back(*outerAttr);
    } else {
      return outer;
    }
  }
}

llvm::Expected<std::vector<ast::InnerAttribute>>
Parser::parseInnerAttributes() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  std::vector<InnerAttribute> inner;

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse inner attributes");
    } else if (checkInnerAttribute()) {
      llvm::Expected<ast::InnerAttribute> innerAttr = parseInnerAttribute();
      if (auto e = innerAttr.takeError()) {
        llvm::errs() << "failed to parse inner attribute in inner attributes: "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      inner.push_back(*innerAttr);
    } else {
      return inner;
    }
  }
}

} // namespace rust_compiler::parser
