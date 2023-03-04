#include "ADT/Result.h"
#include "AST/ExpressionStatement.h"
#include "AST/ItemDeclaration.h"
#include "AST/LetStatement.h"
#include "AST/MacroInvocationSemiStatement.h"
#include "AST/OuterAttribute.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Statement>>
Parser::parseExpressionStatement() {
  Location loc = getLocation();

  llvm::outs() << "parseExpressionStatement"
               << "\n";

  ExpressionStatement exr = {loc};

  if (checkExpressionWithBlock()) {
    StringResult<std::shared_ptr<ast::Expression>> with =
        parseExpressionWithBlock();
    if (!with) {
      llvm::errs() << "failed to expression with block: " << with.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    exr.setExprWoBlock(with.getValue());

    if (check(TokenKind::Semi)) {
      exr.setTrailingSemi();
      assert(eat(TokenKind::Semi));
    }

    return std::make_shared<ExpressionStatement>(exr);
  } else if (checkExpressionWithoutBlock()) {
    llvm::outs() << "parseExpressionStatement: wo"
                 << "\n";
    StringResult<std::shared_ptr<ast::Expression>> wo =
        parseExpressionWithoutBlock();
    if (!wo) {
      llvm::errs() << "failed to expression without block: " << wo.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    exr.setExprWithBlock(wo.getValue());

    if (!check(TokenKind::Semi)) {
      llvm::errs() << "failed to parse ; token in expression statement: "
                   << "\n";
      exit(EXIT_FAILURE);
    }
    assert(eat(TokenKind::Semi));

    return std::make_shared<ExpressionStatement>(exr);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse expression statement");
}

bool Parser::checkStatement() {
  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attribute in check statement: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
  }

  if (checkKeyWord(KeyWordKind::KW_LET))
    return true;
  if (checkExpressionWithBlock())
    return true;
  if (checkExpressionWithoutBlock())
    return true;
  if (checkSimplePathSegment())
    return true;
  if (checkVisItem())
    return true;
  if (checkMacroItem())
    return true;

  return false;
}

StringResult<std::shared_ptr<ast::Statement>>
Parser::parseMacroInvocationSemiStatement() {
  Location loc = getLocation();

  MacroInvocationSemiStatement stmt = {loc};

  StringResult<ast::SimplePath> path = parseSimplePath();
  if (auto e = path.takeError()) {
    llvm::errs()
        << "failed to parse simple path in macro invocation semi statement: "
        << std::move(e) << "\n";
    exit(EXIT_FAILURE);
  }
  stmt.setPath(*path);

  if (!check(TokenKind::Not)) {
    llvm::errs()
        << "failed to parse ! token in macro invocation semi statement: "
        << "\n";
    exit(EXIT_FAILURE);
  }
  assert(eat(TokenKind::Not));

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse macro invocation semi statement: eof");
    } else if (check(TokenKind::ParenOpen)) {
      stmt.setKind(MacroInvocationSemiStatementKind::Paren);
      assert(eat(TokenKind::ParenOpen));
    } else if (check(TokenKind::SquareOpen)) {
      stmt.setKind(MacroInvocationSemiStatementKind::Square);
      assert(eat(TokenKind::SquareOpen));
    } else if (check(TokenKind::BraceOpen)) {
      stmt.setKind(MacroInvocationSemiStatementKind::Brace);
      assert(eat(TokenKind::BraceOpen));
    } else if (check(TokenKind::ParenClose) && check(TokenKind::Semi, 1)) {
      if (stmt.getKind() != MacroInvocationSemiStatementKind::Paren)
        return createStringError(
            inconvertibleErrorCode(),
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::ParenClose));
      assert(eat(TokenKind::Semi));
      return std::make_shared<MacroInvocationSemiStatement>(stmt);
    } else if (check(TokenKind::SquareClose) && check(TokenKind::Semi, 1)) {
      if (stmt.getKind() != MacroInvocationSemiStatementKind::Square)
        return StringResult<std::shared_ptr<ast::Statement>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::SquareClose));
      assert(eat(TokenKind::Semi));
      return std::make_shared<MacroInvocationSemiStatement>(stmt);
    } else if (check(TokenKind::BraceClose)) {
      if (stmt.getKind() != MacroInvocationSemiStatementKind::Brace)
        return StringResult<std::shared_ptr<ast::Statement>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::BraceClose));
      return std::make_shared<MacroInvocationSemiStatement>(stmt);
    } else {
      StringResult<ast::TokenTree> tree = parseTokenTree();
      if (!tree) {
        llvm::errs()
            << "failed to parse token tree in macro invocation statement: "
            << tree.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      stmt.addTree(tree.getValue());
    }
  }

  return StringResult<std::shared_ptr<ast::Statement>>(
      "failed to parse macro invocation semi statement");
}

StringResult<std::shared_ptr<ast::Statement>> Parser::parseLetStatement() {
  Location loc = getLocation();

  LetStatement let = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in let statement: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<OuterAttribute> ot = outer.getValue();
    let.setOuterAttributes(ot);
  }

  if (!checkKeyWord(KeyWordKind::KW_LET)) {
    llvm::errs() << "failed to let token: "
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eatKeyWord(KeyWordKind::KW_LET));

  StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in parse let statement: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  let.setPattern(pattern.getValue());

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    StringResult<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to type in let statement: " << type.getError()
                   << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    let.setType(type.getValue());
  }

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    StringResult<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (!expr) {
      llvm::errs() << "failed to parse expression in let statement: "
                   << expr.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    let.setExpression(expr.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
    StringResult<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (!block) {
      llvm::errs() << "failed to parse block expression in let statement: "
                   << block.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    let.setElseExpr(block.getValue());
  }

  if (!check(TokenKind::Semi)) {
    llvm::errs() << "failed to parse ; token:"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eat(TokenKind::Semi));

  return StringResult<std::shared_ptr<ast::Statement>>(
      std::make_shared<LetStatement>(let));
}

StringResult<std::shared_ptr<ast::Statement>> Parser::parseStatement() {
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();
  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in parse statement: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<OuterAttribute> ot = outer.getValue();
    if (checkVisItem()) {
      StringResult<std::shared_ptr<ast::VisItem>> visItem = parseVisItem(ot);
      if (!visItem) {
        llvm::errs() << "failed to parse vis item in parse statement: "
                     << visItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      std::vector<ast::OuterAttribute> out = outer.getValue();
      ItemDeclaration item = {loc};
      item.setOuterAttributes(out);
      item.setVisItem(visItem.getValue());
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<ItemDeclaration>(item));
    } else if (checkMacroItem()) {
      StringResult<std::shared_ptr<ast::MacroItem>> macroItem =
          parseMacroItem(ot);
      if (!macroItem) {
        llvm::errs() << "failed to parse macro item in parse statement: "
                     << macroItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setOuterAttributes(ot);
      item.setMacroItem(macroItem.getValue());
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<ItemDeclaration>(item));
    } else if (checkKeyWord(KeyWordKind::KW_LET)) {
      recover(cp);
      return parseLetStatement();
    } else {
      return StringResult<std::shared_ptr<ast::Statement>>(
          "failed to parse statement");
    }
  } else { // no outer attributes
    // COPY & PASTE
    std::span<OuterAttribute> outer;
    if (checkVisItem()) {
      StringResult<std::shared_ptr<ast::VisItem>> visItem = parseVisItem(outer);
      if (!visItem) {
        llvm::errs() << "failed to parse vis item in parse statement: "
                     << visItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setVisItem(visItem.getValue());
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<ItemDeclaration>(item));
    } else if (checkMacroItem()) {
      StringResult<std::shared_ptr<ast::MacroItem>> macroItem =
          parseMacroItem(outer);
      if (!macroItem) {
        llvm::errs() << "failed to parse macro item in parse statement: "
                     << macroItem.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setMacroItem(macroItem.getValue());
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<ItemDeclaration>(item));
    } else if (checkKeyWord(KeyWordKind::KW_LET)) {
      return parseLetStatement();
    } else if (checkExpressionWithBlock() || checkExpressionWithoutBlock()) {
      return parseExpressionStatement();
    } else if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
      return parseMacroInvocationSemiStatement();
    } else {
      return StringResult<std::shared_ptr<ast::Statement>>(
          "failed to parse statement");
    }
  }
  return StringResult<std::shared_ptr<ast::Statement>>(
      "failed to parse statement");
}

} // namespace rust_compiler::parser
