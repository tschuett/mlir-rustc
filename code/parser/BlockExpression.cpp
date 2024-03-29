#include "AST/BlockExpression.h"

#include "AST/EmptyStatement.h"
#include "AST/ExpressionStatement.h"
#include "AST/InnerAttribute.h"
#include "AST/ItemDeclaration.h"
#include "AST/OuterAttribute.h"
#include "AST/Statement.h"
#include "AST/Statements.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/ExpressionOrStatement.h"
#include "Parser/Parser.h"
#include "llvm/Support/ErrorHandling.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

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
    return StringResult<ExpressionOrStatement>(
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
    case KeyWordKind::KW_SELFTYPE: {
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

      Result<std::shared_ptr<ast::Expression>, std::string> expr =
          parseExpressionWithoutBlock(outerAttr, restrictions);
      if (!expr) {
        llvm::errs()
            << "failed to parse expression without block in statement or "
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

      break;
    }
    }
  } else if (!getToken().isKeyWord()) {
    switch (getToken().getKind()) {
    case TokenKind::Dollar: {
      if (!checkKeyWord(KeyWordKind::KW_CRATE))
        break;
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
      }
      // must be expression
      return StringResult<ExpressionOrStatement>(
          ExpressionOrStatement(expr.getValue()));
    }
    case TokenKind::BraceOpen: {
      // maybe block expr
      CheckPoint cp = getCheckPoint();
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
                          "failed to parse block expression in statement or "
                          "expression without block",
                          block.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }

      if (check(TokenKind::BraceClose) or check(TokenKind::Semi)) {
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(block.getValue()));
      } else {
        recover(cp);
        adt::Result<std::shared_ptr<ast::Expression>, std::string> expr =
            parseExpression({}, restrictions);
        if (!expr) {
          // report error
          llvm::errs()
              << "failed to parse expression without block in statement or "
                 "expression without block: "
              << expr.getError() << "\n";
          // report error
          std::string s =
              llvm::formatv(
                  "{0}\n{1}",
                  "failed to parse expression without block  in statement or "
                  "expression without block",
                  expr.getError())
                  .str();
          return StringResult<ExpressionOrStatement>(s);
        }
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(expr.getValue()));
      }
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
    case TokenKind::CHAR_LITERAL:
    case TokenKind::STRING_LITERAL:
    case TokenKind::RAW_STRING_LITERAL:
    case TokenKind::BYTE_LITERAL:
    case TokenKind::BYTE_STRING_LITERAL:
    case TokenKind::RAW_BYTE_STRING_LITERAL:
    case TokenKind::INTEGER_LITERAL:
    case TokenKind::FLOAT_LITERAL: {
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
    case TokenKind::ParenOpen: {
      adt::Result<std::shared_ptr<ast::Expression>, std::string> paren =
          parseGroupedOrTupleExpression(restrictions);
      if (!paren) {
        llvm::errs() << "failed to parse grouped or parenthesis expression in "
                        "statement or expression without block"
                     << "\n";
        std::string s =
            llvm::formatv(
                "failed to parse grouped or parenthesis expression in "
                "statement or expression without block: {0}",
                paren.getError())
                .str();
        return StringResult<ExpressionOrStatement>(s);
      }
      if (getToken().getKind() == TokenKind::Semi) {
        assert(eat(TokenKind::Semi));
        // eat and expression statement
        ExpressionStatement stmt = {getLocation()};
        stmt.setExprWoBlock(paren.getValue());
        stmt.setTrailingSemi();
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(std::make_shared<ExpressionStatement>(stmt)));
      } else {
        return StringResult<ExpressionOrStatement>(
            ExpressionOrStatement(paren.getValue()));
      }
    }
    case TokenKind::Lt: {
      // qualified path and ...
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

      Result<std::shared_ptr<ast::Expression>, std::string> expr =
          parseExpressionWithoutBlock(outerAttr, restrictions);
      if (!expr) {
        llvm::errs()
            << "failed to parse expression without block in statement or "
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
    }
    }
  }
  llvm_unreachable("either keyword or not");
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseBlockExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  BlockExpression bloc = {loc};

  //  llvm::errs() << "parseBlockExpression2"
  //               << "\n";

  if (!check(TokenKind::BraceOpen)) {
    llvm::errs() << Token2String(getToken().getKind()) << "\n";
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
    std::vector<InnerAttribute> inner = innerAttributes.getValue();
    bloc.setInnerAttributes(inner);
  }

  Statements stmts = {loc};

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    bloc.setStatements(stmts);

    return Result<std::shared_ptr<ast::Expression>, std::string>(
        std::make_shared<BlockExpression>(bloc));
  }

  while (getToken().getKind() != TokenKind::BraceClose) {
    adt::StringResult<ExpressionOrStatement> expr =
        parseStatementOrExpressionWithoutBlock();
    if (!expr) {
      // report error
      llvm::errs() << "failed to parse statement or expression without block "
                      "in block expression: "
                   << expr.getError() << "\n";
      std::string s = llvm::formatv("{0}\n{1}",
                                    "failed to parse statement or expression "
                                    "without block in block expression : ",
                                    expr.getError())
                          .str();
      return Result<std::shared_ptr<ast::Expression>, std::string>(s);
    }

    ExpressionOrStatement eos = expr.getValue();

    switch (eos.getKind()) {
    case ExpressionOrStatementKind::Expression: {
      stmts.setTrailing(eos.getExpression());
      break;
    }
    case ExpressionOrStatementKind::Statement: {
      stmts.addStmt(eos.getStatement());
      break;
    }
    case ExpressionOrStatementKind::Item: {
      std::shared_ptr<ast::Item> item = eos.getItem();
      ItemDeclaration decl = {item->getLocation()};
      decl.setVisItem(item);
      stmts.addStmt(std::make_shared<ItemDeclaration>(decl));
      break;
    }
    }
    // if (eos.getKind() == ExpressionOrStatementKind::Statement) {
    //   stmts.addStmt(eos.getStatement());
    // } else if (eos.getKind() == ExpressionOrStatementKind::Item) {
    //   std::shared_ptr<ast::Item> item = eos.getItem();
    //   ItemDeclaration decl = {item->getLocation()};
    //   decl.setVisItem(item);
    //   stmts.addStmt(std::make_shared<ItemDeclaration>(decl));
    // } else if (eos.getKind() == ExpressionOrStatementKind::Expression) {
    //   stmts.setTrailing(eos.getExpression());
    // } else {
    //   llvm::er break;
    // }
  }

  if (!check(TokenKind::BraceClose)) {
    llvm::errs() << "real token: " << Token2String(getToken().getKind())
                 << "\n";
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse } in block expression");
  }
  assert(eat(TokenKind::BraceClose));

  bloc.setStatements(stmts);

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<BlockExpression>(bloc));
}

} // namespace rust_compiler::parser
