#include "AST/ItemDeclaration.h"
#include "AST/LetStatement.h"
#include "AST/MacroInvocationSemi.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include <llvm/Support/raw_ostream.h>
#include "AST/MacroInvocationSemiStatement.h"

#include <cstdlib>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Statement>>
Parser::parseMacroInvocationSemiStatement() {
  Location loc = getLocation();

  MacroInvocationSemiStatement stmt = {loc};

  llvm::Expected<ast::SimplePath> path = parseSimplePath();
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

  llvm::Expected<ast::DelimTokenTree> tree = parseDelimTokenTree();
  if (auto e = tree.takeError()) {
    llvm::errs()
        << "failed to parse delim token tree  macro invocation semi statement: "
        << std::move(e) << "\n";
    exit(EXIT_FAILURE);
  }
  stmt.setTree(*tree);

  return std::make_shared<MacroInvocationSemiStatement>(stmt);
}

llvm::Expected<std::shared_ptr<ast::Statement>> Parser::parseLetStatement() {
  Location loc = getLocation();

  LetStatement let = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes: " << std::move(e)
                   << "\n";
      exit(EXIT_FAILURE);
    }
    let.setOuterAttributes(*outer);
  }

  if (!checkKeyWord(KeyWordKind::KW_LET)) {
    llvm::errs() << "failed to let token: "
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eatKeyWord(KeyWordKind::KW_LET));

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt: " << std::move(e)
                 << "\n";
    exit(EXIT_FAILURE);
  }
  let.setPattern(*pattern);

  if (check(TokenKind::Colon)) {
    assert(eat(TokenKind::Colon));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type: " << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    let.setType(*type);
  }

  if (check(TokenKind::Eq)) {
    assert(eat(TokenKind::Eq));
    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression: " << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    let.setExpression(*expr);
  }

  if (checkKeyWord(KeyWordKind::KW_ELSE)) {
    assert(eatKeyWord(KeyWordKind::KW_ELSE));
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (auto e = block.takeError()) {
      llvm::errs() << "failed to parse block expression: " << std::move(e)
                   << "\n";
      exit(EXIT_FAILURE);
    }
    let.setElseExpr(*block);
  }

  if (!check(TokenKind::Semi)) {
    llvm::errs() << "failed to parse ; token:"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eat(TokenKind::Semi));

  return std::make_shared<LetStatement>(let);
}

llvm::Expected<std::shared_ptr<ast::Statement>> Parser::parseStatement() {
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();
  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in statement: "
                   << std::move(e) << "\n";
      exit(EXIT_FAILURE);
    }
    if (checkVisItem()) {
      llvm::Expected<std::shared_ptr<ast::VisItem>> visItem = parseVisItem();
      if (auto e = visItem.takeError()) {
        llvm::errs() << "failed to parse VisItem in statement: " << std::move(e)
                     << "\n";
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setOuterAttributes(*outer);
      item.setVisItem(*visItem);
      return std::make_shared<ItemDeclaration>(item);
    } else if (checkMacroItem()) {
      llvm::Expected<std::shared_ptr<ast::MacroItem>> macroItem =
          parseMacroItem();
      if (auto e = macroItem.takeError()) {
        llvm::errs() << "failed to parse MacroItem in statement: "
                     << std::move(e) << "\n";
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setOuterAttributes(*outer);
      item.setMacroItem(*macroItem);
      return std::make_shared<ItemDeclaration>(item);
    } else if (checkKeyWord(KeyWordKind::KW_LET)) {
      recover(cp);
      return parseLetStatement();
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse statement");
    }
  } else { // no outer attributes
    // COPY & PASTE
    if (checkVisItem()) {
      llvm::Expected<std::shared_ptr<ast::VisItem>> visItem = parseVisItem();
      if (auto e = visItem.takeError()) {
        llvm::errs() << "failed to parse VisItem in statement: " << std::move(e)
                     << "\n";
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setVisItem(*visItem);
      return std::make_shared<ItemDeclaration>(item);
    } else if (checkMacroItem()) {
      llvm::Expected<std::shared_ptr<ast::MacroItem>> macroItem =
          parseMacroItem();
      if (auto e = macroItem.takeError()) {
        llvm::errs() << "failed to parse MacroItem in statement: "
                     << std::move(e) << "\n";
        exit(EXIT_FAILURE);
      }
      ItemDeclaration item = {loc};
      item.setMacroItem(*macroItem);
      return std::make_shared<ItemDeclaration>(item);
    } else if (checkKeyWord(KeyWordKind::KW_LET)) {
      return parseLetStatement();
    } else if (checkExpressionWithBlock() || checkExpressionWithoutBlock()) {
      return parseExpressionStatement();
    } else if (check(TokenKind::PathSep) || checkSimplePathSegment()) {
      return parseMacroInvocationSemiStatement();
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse statement");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse statement");
}

} // namespace rust_compiler::parser
