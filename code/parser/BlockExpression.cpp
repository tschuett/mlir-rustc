#include "AST/BlockExpression.h"

#include "AST/EmptyStatement.h"
#include "AST/ExpressionStatement.h"
#include "AST/OuterAttribute.h"
#include "AST/Statement.h"
#include "AST/Statements.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/ExpressionOrStatement.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

StringResult<ExpressionOrStatement>
Parser::parseStatementOrExpressionWithBlock(std::span<OuterAttribute> outer) {
  Location loc = getLocation();

  Result<std::shared_ptr<ast::Expression>, std::string> withBlock =
      parseExpressionWithBlock(outer);
  if (!withBlock) {
    // report error
    llvm::errs() << "failed to parse expression with block in statement or "
                    "expression with block: "
                 << withBlock.getError() << "\n";
    std::string s =
        llvm::formatv("{0}\n{1}",
                      "failed to parse expression with block in statement or "
                      "expression with block",
                      withBlock.getError())
            .str();
    return StringResult<ExpressionOrStatement>(s);
  }

  if (getToken().getKind() == TokenKind::BraceClose) {
    // tail expression in block expression
    return StringResult<ExpressionOrStatement>(
        ExpressionOrStatement(withBlock.getValue()));
  }

  ExpressionStatement stmt = {loc};
  stmt.setExprWithBlock(withBlock.getValue());
  if (getToken().getKind() == TokenKind::Semi) {
    stmt.setTrailingSemi();
    assert(eat(TokenKind::Semi));
  }

  return StringResult<ExpressionOrStatement>(
      ExpressionOrStatement(std::make_shared<ExpressionStatement>(stmt)));
}

StringResult<ExpressionOrStatement>
Parser::parseStatementOrExpressionWithoutBlock() {
  Location loc = getLocation();

  if (getToken().getKind() == TokenKind::Semi) {
    // empty statement; early exit
    assert(eat(TokenKind::Semi));
    StringResult<ExpressionOrStatement>(
        ExpressionOrStatement(std::make_shared<EmptyStatement>(loc)));
  }

  std::vector<ast::OuterAttribute> outerAttr;

  // parse outer attr
  if (checkOuterAttribute()) {
    Result<std::vector<ast::OuterAttribute>, std::string> outer =
        parseOuterAttributes();
    if (!outer) {
      // report error
      llvm::errs() << "failed to parse outer attributes in statement or "
                      "expression without block: "
                   << outer.getError() << "\n";
      std::string s =
          llvm::formatv("{0}\n{1}",
                        "failed to parse outer attributes in statement or "
                        "expression without block",
                        outer.getError())
              .str();
      return StringResult<ExpressionOrStatement>(s);
    }
    outerAttr = outer.getValue();
  }

  Restrictions restrictions;

  if (getToken().isKeyWord()) {
    switch (getToken().getKeyWordKind()) {
    case KeyWordKind::KW_LET: {
      // let statement
      Result<std::shared_ptr<ast::Statement>, std::string> let =
          parseLetStatement(outerAttr, restrictions);
      if (!let) {
        llvm::errs() << "failed to parse let statement in statement or "
                        "expression without block: "
                     << let.getError() << "\n";
        std::string s =
            llvm::formatv("{0}\n{1}",
                          "failed to parse let statement in statement or "
                          "expression without block: ",
                          let.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }
      return StringResult<ExpressionOrStatement>(
          ExpressionOrStatement(let.getValue()));
    }
    case KeyWordKind::KW_PUB:
    case KeyWordKind::KW_MOD:
    case KeyWordKind::KW_EXTERN:
    case KeyWordKind::KW_USE:
    case KeyWordKind::KW_FN:
    case KeyWordKind::KW_TYPE:
    case KeyWordKind::KW_STRUCT:
    case KeyWordKind::KW_ENUM:
    case KeyWordKind::KW_CONST:
    case KeyWordKind::KW_STATIC:
    case KeyWordKind::KW_TRAIT:
    case KeyWordKind::KW_UNION:
    case KeyWordKind::KW_IMPL: {
      // parse vis item
      Result<std::shared_ptr<ast::Item>, std::string> item =
          parseVisItem(outerAttr);
      if (!item) {
        llvm::errs() << "failed to parse vis item in statement or "
                        "expression without block: "
                     << item.getError() << "\n";
        std::string s =
            llvm::formatv("{0}\n{1}",
                          "failed to parse vis item in statement or "
                          "expression without block: ",
                          item.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      };
      return StringResult<ExpressionOrStatement>(
          ExpressionOrStatement(item.getValue()));
    }
    case KeyWordKind::KW_UNSAFE: {
      if (check(TokenKind::BraceOpen, 1)) {
        // unsafe block
        return parseStatementOrExpressionWithBlock(outerAttr);
      } else if (checkKeyWord(KeyWordKind::KW_TRAIT, 1)) {
        // unsafe trait
        Result<std::shared_ptr<ast::Item>, std::string> item =
            parseVisItem(outerAttr);
        if (!item) {
          llvm::errs() << "failed to parse vis item in statement or "
                          "expression without block: "
                       << item.getError() << "\n";
          std::string s =
              llvm::formatv("{0}\n{1}",
                            "failed to parse vis item in statement or "
                            "expression without block: ",
                            item.getError())
                  .str();
          return StringResult<ExpressionOrStatement>(s);
        }
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(item.getValue()));
      } else if (checkKeyWord(KeyWordKind::KW_FN, 1)) {
        // unsafe function
        Result<std::shared_ptr<ast::Item>, std::string> item =
            parseVisItem(outerAttr);
        if (!item) {
          llvm::errs() << "failed to parse vis item in statement or "
                          "expression without block: "
                       << item.getError() << "\n";
          std::string s =
              llvm::formatv("{0}\n{1}",
                            "failed to parse vis item in statement or "
                            "expression without block: ",
                            item.getError())
                  .str();
        }
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(item.getValue()));
      } else if (checkKeyWord(KeyWordKind::KW_IMPL, 1)) {
        // unsafe trait impl
        Result<std::shared_ptr<ast::Item>, std::string> item =
            parseVisItem(outerAttr);
        if (!item) {
          llvm::errs() << "failed to parse vis item in statement or "
                          "expression without block: "
                       << item.getError() << "\n";
          std::string s =
              llvm::formatv("{0}\n{1}",
                            "failed to parse vis item in statement or "
                            "expression without block: ",
                            item.getError())
                  .str();
        }
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(item.getValue()));
      } else {
        // report error after unsafe keyword
        llvm::errs() << "unknown kind of unsafe structure in statement or "
                        "expression without block"
                     << "\n";
        return StringResult<ExpressionOrStatement>(
            "unknown kind of unsafe structure in statement or "
            "expression without block");
      }
    }
    case KeyWordKind::KW_SUPER:
    case KeyWordKind::KW_SELFVALUE:
    case KeyWordKind::KW_SELFTYPE:
    case KeyWordKind::KW_DOLLARCRATE: {
      // something that starts with a path:
      Restrictions restrictions;
      Result<std::shared_ptr<ast::Expression>, std::string> expr =
          parseExpressionWithoutBlock(outerAttr, restrictions);
      if (!expr) {
        llvm::errs() << "failed to parse expression without block in statement "
                        "or expression without block"
                     << "\n";
      }
      if (check(TokenKind::Semi)) {
        // must be expression statement
        assert(eat(TokenKind::Semi));
        ExpressionStatement stmt = {loc};
        stmt.setTrailingSemi();
        stmt.setExprWoBlock(expr.getValue());
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(std::make_shared<ExpressionStatement>(stmt)));
      } else {
        // must be expression
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(expr.getValue()));
      }
    }
    case KeyWordKind::KW_LOOP:
    case KeyWordKind::KW_WHILE:
    case KeyWordKind::KW_FOR:
    case KeyWordKind::KW_IF:
    case KeyWordKind::KW_MATCH:
    case KeyWordKind::KW_ASYNC: {
      return StringResult<ExpressionOrStatement>(
          parseStatementOrExpressionWithBlock(outerAttr));
    }
    case KeyWordKind::KW_MACRO_RULES: {
      /// macro_rules!
      adt::Result<std::shared_ptr<ast::Item>, std::string> macro =
          parseMacroRulesDefinition();
      if (!macro) {
        // report error
        llvm::errs() << "failed to parse macro rules definition in expression "
                        "or statement: "
                     << macro.getError() << "\n";
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse macro rules definition in expression "
                "or statement",
                macro.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }
      return StringResult<ExpressionOrStatement>(
          ExpressionOrStatement(macro.getValue()));
    }
    default: {
      llvm::errs()
          << "unknown keyword in statement or expression without block: "
          << KeyWord2String(getToken().getKeyWordKind()) << "\n";
      break;
    }
    }
  } else if (!getToken().isKeyWord()) {
    switch (getToken().getKind()) {
    case TokenKind::BraceOpen: {
      // block expr
      Result<std::shared_ptr<ast::Expression>, std::string> block =
          parseBlockExpression(outerAttr);
      if (!block) {
        // report error
        llvm::errs() << "failed to parse block expression in statement or "
                        "expression without block: "
                     << block.getError() << "\n";
        // report error
        std::string s =
            llvm::formatv("{0}\n{1}",
                          "failed to parse bloc kexpression in statement or "
                          "expression without block",
                          block.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }
      return StringResult<ExpressionOrStatement>(
          ExpressionOrStatement(block.getValue()));
    }
    case TokenKind::Identifier: {
      // path and ...
      Result<std::shared_ptr<ast::Expression>, std::string> woBlock =
          parseExpressionWithoutBlock(outerAttr, restrictions);
      if (!woBlock) {
        // report block
        llvm::errs()
            << "failed to parse expression without block in statement or "
               "expression without block: "
            << woBlock.getError() << "\n";
        // report error
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse expression without block in statement or "
                "expression without block",
                woBlock.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }

      if (getToken().getKind() == TokenKind::Semi) {
        assert(eat(TokenKind::Semi));
        // eat and expression statement
        ExpressionStatement stmt = {getLocation()};
        stmt.setExprWoBlock(woBlock.getValue());
        stmt.setTrailingSemi();
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(std::make_shared<ExpressionStatement>(stmt)));
      } else {
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(woBlock.getValue()));
      }
    }
    default: {
      llvm::errs() << "unknown token in statement or expression without block: "
                   << Token2String(getToken().getKind()) << "\n";
    }
    }
  }

  Result<std::shared_ptr<ast::Expression>, std::string> expr =
      parseExpressionWithoutBlock(outerAttr, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression without block in statement or "
                    "expression without block: "
                 << expr.getError() << "\n";
    // report error
    std::string s =
        llvm::formatv(
            "{0}\n{1}",
            "failed to parse expression without block in statement or "
            "expression without block",
            expr.getError())
            .str();
    return StringResult<ExpressionOrStatement>(s);
  }

  if (getToken().getKind() == TokenKind::Semi) {
    assert(eat(TokenKind::Semi));
    // found expression statement
    ExpressionStatement stmt = {getLocation()};
    stmt.setExprWoBlock(expr.getValue());
    stmt.setTrailingSemi();
    return StringResult<ExpressionOrStatement>(
        ExpressionOrStatement(std::make_shared<ExpressionStatement>(stmt)));
  }

  // expression
  return StringResult<ExpressionOrStatement>(
      ExpressionOrStatement(expr.getValue()));

  // FIXME lifetimes
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseBlockExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  BlockExpression bloc = {loc};

  llvm::errs() << "parseBlockExpression"
               << "\n";

  if (!check(TokenKind::BraceOpen)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse { in block expression");
  }

  assert(eat(TokenKind::BraceOpen));

  if (checkInnerAttribute()) {
    StringResult<std::vector<ast::InnerAttribute>> innerAttributes =
        parseInnerAttributes();
    if (!innerAttributes) {
      llvm::errs() << "failed to parse inner attributes in block expression: "
                   << innerAttributes.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0}\n{1}",
                        "failed to parse inner attributes in block expression",
                        innerAttributes.getError())
              .str();
      return Result<std::shared_ptr<ast::Expression>, std::string>(s);
    }
  }

  Statements stmts = {loc};

  while (getToken().getKind() != TokenKind::BraceClose) {
    adt::StringResult<ExpressionOrStatement> expr =
        parseStatementOrExpressionWithoutBlock();
    if (!expr) {
      // report error
    }

    if (expr.getValue().getKind() == ExpressionOrStatementKind::Statement) {
      stmts.addStmt(expr.getValue().getStatement());
    } else {
      stmts.setTrailing(expr.getValue().getExpression());
      break;
    }
  }

  if (!check(TokenKind::BraceClose)) {
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse } in block expression");
  }
  assert(eat(TokenKind::BraceClose));

  BlockExpression block = {loc};
  bloc.setStatements(stmts);

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<BlockExpression>(block));
}

} // namespace rust_compiler::parser
