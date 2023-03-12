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
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::Statement>>
Parser::parseExpressionStatement(std::span<ast::OuterAttribute> outer,
                                 Restrictions restrictions) {
  Location loc = getLocation();

  llvm::outs() << "parseExpressionStatement"
               << "\n";

  ExpressionStatement exr = {loc};

  if (checkExpressionWithBlock()) {
    StringResult<std::shared_ptr<ast::Expression>> with =
        parseExpressionWithBlock(outer);
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

    return StringResult<std::shared_ptr<ast::Statement>>(
        std::make_shared<ExpressionStatement>(exr));
  } else if (checkExpressionWithoutBlock()) {
    llvm::outs() << "parseExpressionStatement: wo"
                 << "\n";
    StringResult<std::shared_ptr<ast::Expression>> wo =
        parseExpressionWithoutBlock(outer, restrictions);
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

    return StringResult<std::shared_ptr<ast::Statement>>(
        std::make_shared<ExpressionStatement>(exr));
  }
  return StringResult<std::shared_ptr<ast::Statement>>(
      "failed to parse expression statement");
}

bool Parser::checkStatement() {
  llvm::errs() << "checkStatement"
               << "\n";

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
  if (checkKeyWord(KeyWordKind::KW_RETURN))
    return false;
  if (checkKeyWord(KeyWordKind::KW_CONST))
    return true;
  if (checkKeyWord(KeyWordKind::KW_STATIC))
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
  if (!path) {
    llvm::errs() << "failed to simple path in macro invocation statement: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  stmt.setPath(path.getValue());

  if (!check(TokenKind::Not)) {
    llvm::errs()
        << "failed to parse ! token in macro invocation semi statement: "
        << "\n";
    exit(EXIT_FAILURE);
  }
  assert(eat(TokenKind::Not));

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<std::shared_ptr<ast::Statement>>(
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
        return StringResult<std::shared_ptr<ast::Statement>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::ParenClose));
      assert(eat(TokenKind::Semi));
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<MacroInvocationSemiStatement>(stmt));
    } else if (check(TokenKind::SquareClose) && check(TokenKind::Semi, 1)) {
      if (stmt.getKind() != MacroInvocationSemiStatementKind::Square)
        return StringResult<std::shared_ptr<ast::Statement>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::SquareClose));
      assert(eat(TokenKind::Semi));
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<MacroInvocationSemiStatement>(stmt));
    } else if (check(TokenKind::BraceClose)) {
      if (stmt.getKind() != MacroInvocationSemiStatementKind::Brace)
        return StringResult<std::shared_ptr<ast::Statement>>(
            "failed to parse macro invocation semi statement");
      assert(eat(TokenKind::BraceClose));
      return StringResult<std::shared_ptr<ast::Statement>>(
          std::make_shared<MacroInvocationSemiStatement>(stmt));
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

StringResult<std::shared_ptr<ast::Statement>>
Parser::parseLetStatement(std::span<ast::OuterAttribute> outer,
                          Restrictions restrictions) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  LetStatement let = {loc};

  llvm::errs() << "parse let statement"
               << "\n";

  let.setOuterAttributes(outer);

  if (!checkKeyWord(KeyWordKind::KW_LET)) {
    llvm::errs() << "failed to let token: "
                 << "\n";
    std::string s = llvm::formatv("{0}", "failed to parse let token").str();
    // exit(EXIT_FAILURE);
    return StringResult<std::shared_ptr<ast::Statement>>(s);
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
      llvm::errs() << "failed to parse type in let statement: "
                   << type.getError() << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv("{0} {1}", "failed to parse type in let statement: ",
                        type.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Statement>>(s);
    }
    let.setType(type.getValue());
  }

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    StringResult<std::shared_ptr<ast::Expression>> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in let statement: "
                   << expr.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv(
              "{0} {1}",
              "failed to parse expression in let statement: ", expr.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Statement>>(s);
      // exit(EXIT_FAILURE);
    }
    let.setExpression(expr.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
    StringResult<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression({});
    if (!block) {
      llvm::errs() << "failed to parse block expression in let statement: "
                   << block.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse block expression in let statement: ",
                        block.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Statement>>(s);
      // exit(EXIT_FAILURE);
    }
    let.setElseExpr(block.getValue());
  }

  if (!check(TokenKind::Semi)) {
    llvm::errs() << "let statement; failed to parse ; token:"
                 << "\n";
    return StringResult<std::shared_ptr<ast::Statement>>(
        "let statement: failed to parse ; token:");
    // exit(EXIT_FAILURE);
  }

  assert(eat(TokenKind::Semi));

  return StringResult<std::shared_ptr<ast::Statement>>(
      std::make_shared<LetStatement>(let));
}

StringResult<std::shared_ptr<ast::Statement>>
Parser::parseStatement(Restrictions restriction) {
  llvm::errs() << "parse statement"
               << "\n";

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
      StringResult<std::shared_ptr<ast::Item>> visItem = parseVisItem(ot);
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
      StringResult<std::shared_ptr<ast::Item>> macroItem = parseMacroItem(ot);
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
      return parseLetStatement(ot, restriction);
    } else {
      return StringResult<std::shared_ptr<ast::Statement>>(
          "failed to parse statement");
    }
  } else { // no outer attributes
    // COPY & PASTE
    std::span<OuterAttribute> outer;
    if (checkVisItem()) {
      StringResult<std::shared_ptr<ast::Item>> visItem = parseVisItem(outer);
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
      llvm::errs() << "parse statement: macro item"
                   << "\n";
      StringResult<std::shared_ptr<ast::Item>> macroItem =
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
      return parseLetStatement(outer, restriction);
    } else if (checkExpressionWithBlock() || checkExpressionWithoutBlock()) {
      llvm::errs() << "parse statement: expr stmt"
                   << "\n";
      return parseExpressionStatement(outer, restriction);
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
