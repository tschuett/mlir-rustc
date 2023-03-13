#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

StringResult<ast::patterns::StructPatternFields>
Parser::parseStructPatternFields() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  //  llvm::errs() << "parseStructPatternFields"
  //               << "\n";
  //  llvm::errs() << "parseStructPatternFields: "
  //               << Token2String(getToken().getKind()) << "\n";

  StructPatternFields fields = {loc};

  StringResult<ast::patterns::StructPatternField> first =
      parseStructPatternField();
  if (!first) {
    llvm::errs() << "failed to parse struct pattern field in "
                    "parse struct pattern fields: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse first struct pattern field in "
                      "parse struct pattern fields: ",
                      first.getError())
            .str();
    return StringResult<ast::patterns::StructPatternFields>(s);
  }
  fields.addPattern(first.getValue());

  while (true) {
    //    llvm::errs() << "parseStructPatternfields loop: "
    //                 << Token2String(getToken().getKind()) << "\n";
    if (check(TokenKind::Eof)) {
      return StringResult<ast::patterns::StructPatternFields>(
          "failed to parse struct pattern fields: eof");
    } else if (check(TokenKind::Comma)) {
      assert(eat((TokenKind::Comma)));
      StringResult<ast::patterns::StructPatternField> next =
          parseStructPatternField();
      if (!next) {
        std::string s =
            llvm::formatv("{0} {1}",
                          "failed to parse next struct pattern field in "
                          "parse struct pattern fields: ",
                          next.getError())
                .str();
        llvm::errs() << "failed to parse next struct pattern field in "
                        "parse struct pattern fields: "
                     << next.getError() << "\n";
        printFunctionStack();
        return StringResult<ast::patterns::StructPatternFields>(s);
      }
      fields.addPattern(next.getValue());
    } else if (check(TokenKind::BraceClose)) {
      // done
      return StringResult<ast::patterns::StructPatternFields>(fields);
    } else {
      // error
      return StringResult<ast::patterns::StructPatternFields>(fields);
    }
  }

  return StringResult<ast::patterns::StructPatternFields>(
      "failed to parse struct pattern fields");
}

StringResult<ast::patterns::StructPatternField>
Parser::parseStructPatternField() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  //  llvm::errs() << "parseStructPatternField"
  //               << "\n";
  //  llvm::errs() << "parseStructPatternField: "
  //               << Token2String(getToken().getKind()) << "\n";

  StructPatternField field = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse  outer attributes in "
                      "parse struct pattern field: "
                   << outer.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse  outer attributes in "
                                    "parse struct pattern field: ",
                                    outer.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternField>(s);
    }
    std::vector<OuterAttribute> out = outer.getValue();
    field.setOuterAttributes(out);
  }

  if (check(TokenKind::INTEGER_LITERAL) && check(TokenKind::Colon, 1)) {
    field.setTupleIndex(getToken().getLiteral());
    assert(eat(TokenKind::INTEGER_LITERAL));
    assert(eat(TokenKind::Colon));
    StringResult<std::shared_ptr<ast::patterns::Pattern>> patterns =
        parsePattern();
    if (!patterns) {
      llvm::errs() << "failed to parse  pattern in "
                      "parse struct pattern field: "
                   << patterns.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse  pattern in "
                                    "parse struct pattern field: ",
                                    patterns.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternField>(s);
    }
    field.setPattern(patterns.getValue());
    field.setKind(StructPatternFieldKind::TupleIndex);
    return StringResult<ast::patterns::StructPatternField>(field);
  } else if (checkIdentifier() && check(TokenKind::Colon, 1)) {
    field.setIdentifier(getToken().getIdentifier());
    assert(eat(TokenKind::Identifier));
    assert(eat(TokenKind::Colon));
    StringResult<std::shared_ptr<ast::patterns::Pattern>> patterns =
        parsePattern();
    if (!patterns) {
      llvm::errs() << "failed to parse  pattern in "
                      "parse struct pattern field: "
                   << patterns.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse  pattern in "
                                    "parse struct pattern field: ",
                                    patterns.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternField>(s);
    }
    field.setPattern(patterns.getValue());
    field.setKind(StructPatternFieldKind::Identifier);
    return StringResult<ast::patterns::StructPatternField>(field);
  } else if (checkKeyWord(KeyWordKind::KW_REF) ||
             checkKeyWord(KeyWordKind::KW_MUT) || checkIdentifier()) {
    if (checkKeyWord(KeyWordKind::KW_REF)) {
      field.setRef();
      assert(eatKeyWord(KeyWordKind::KW_REF));
    }
    if (checkKeyWord(KeyWordKind::KW_MUT)) {
      field.setMut();
      assert(eatKeyWord(KeyWordKind::KW_MUT));
    }
    if (!checkIdentifier())
      return StringResult<ast::patterns::StructPatternField>(
          "failed to parse identifier token in struct pattern field");
    field.setIdentifier(getToken().getIdentifier());
    field.setKind(StructPatternFieldKind::RefMut);
    assert(eat(TokenKind::Identifier));
    return StringResult<ast::patterns::StructPatternField>(field);
  } // else {
    // return StringResult<ast::patterns::StructPatternField>(
    //     "failed to parse struct pattern field");
  //} //

  llvm::errs() << "parseStructPatternField: "
               << Token2String(getToken().getKind()) << "\n";

  return StringResult<ast::patterns::StructPatternField>(
      "failed to parse struct pattern field");
}

StringResult<ast::patterns::StructPatternEtCetera>
Parser::parseStructPatternEtCetera() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  StructPatternEtCetera et = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in "
                      "parse struct pattern etcetera  pattern: "
                   << outer.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse outer attributes in "
                                    "parse struct pattern etcetera  pattern: ",
                                    outer.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternEtCetera>(s);
    }
    std::vector<ast::OuterAttribute> out = outer.getValue();
    et.setOuterAttributes(out);
  }

  if (!check(TokenKind::DotDot))
    return StringResult<ast::patterns::StructPatternEtCetera>(
        "failed to parse struct pattern etcetera");
  assert(eat(TokenKind::DotDot));

  return StringResult<ast::patterns::StructPatternEtCetera>(et);
}

StringResult<ast::patterns::StructPatternElements>
Parser::parseStructPatternElements() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  StructPatternElements elements = {loc};

  CheckPoint cp = getCheckPoint();

  llvm::errs() << "parseStructPatternElements"
               << "\n";

  llvm::errs() << "parseStructPatternElements: "
               << Token2String(getToken().getKind()) << "\n";

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in "
                      "parse struct elements: "
                   << outer.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse outer attributes in "
                                    "parse struct elements: ",
                                    outer.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternElements>(s);
    }
  }

  if (check(TokenKind::DotDot)) {
    // StructPatternEtCetera
    recover(cp);
    StringResult<ast::patterns::StructPatternEtCetera> etcetera =
        parseStructPatternEtCetera();
    if (!etcetera) {
      llvm::errs() << "failed to parse struct pattern etcetera in "
                      "parse struct elements: "
                   << etcetera.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse struct pattern etcetera in "
                        "parse struct elements: ",
                        etcetera.getError())
              .str();
      return StringResult<ast::patterns::StructPatternElements>(s);
    }
    elements.setEtCetera(etcetera.getValue());
    return StringResult<ast::patterns::StructPatternElements>(elements);
  } else {
    // StructPatternFields
    recover(cp);
    StringResult<ast::patterns::StructPatternFields> fields =
        parseStructPatternFields();
    if (!fields) {
      llvm::errs() << "failed to parse struct pattern fields in "
                      "parse struct elements: "
                   << fields.getError() << "\n";
      printFunctionStack();
      std::string s = llvm::formatv("{0} {1}",
                                    "failed to parse struct pattern fields in "
                                    "parse struct elements: ",
                                    fields.getError())
                          .str();
      return StringResult<ast::patterns::StructPatternElements>(s);
    }
    elements.setFields(fields.getValue());

    if (check(TokenKind::BraceClose))
      return StringResult<ast::patterns::StructPatternElements>(elements);
    if (check(TokenKind::Comma) && check(TokenKind::BraceClose, 1)) {
      assert(eat(TokenKind::Comma));
      return StringResult<ast::patterns::StructPatternElements>(elements);
    }

    StringResult<ast::patterns::StructPatternEtCetera> etcetera =
        parseStructPatternEtCetera();
    if (!etcetera) {
      llvm::errs() << "failed to parse struct pattern etcetera in "
                      "parse struct elements: "
                   << etcetera.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse struct pattern etcetera in "
                        "parse struct elements: ",
                        etcetera.getError())
              .str();
      return StringResult<ast::patterns::StructPatternElements>(s);
    }
    elements.setEtCetera(etcetera.getValue());
    return StringResult<ast::patterns::StructPatternElements>(elements);
  }

  llvm::errs() << "parseStructPatternElements: "
               << Token2String(getToken().getKind()) << "\n";

  return StringResult<ast::patterns::StructPatternElements>(
      "failed to parse struct pattern elements");
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseStructPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  //  llvm::errs() << "parseStructPattern"
  //               << "\n";

  StructPattern pat = {loc};

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::errs() << "failed to parse path in expression in "
                    "parse struct pattern: "
                 << path.getError() << "\n";
    printFunctionStack();
    std::string s = llvm::formatv("{0} {1}",
                                  "failed to parse path in expression in "
                                  "parse struct pattern: ",
                                  path.getError())
                        .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
  }
  pat.setPath(path.getValue());
  if (!check(lexer::TokenKind::BraceOpen)) {
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse struct pattern");
  }
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<StructPattern>(pat));
  }

  StringResult<StructPatternElements> pattern = parseStructPatternElements();
  if (!pattern) {
    llvm::errs() << "failed to parse struct pattern elements in "
                    "parse struct pattern: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    std::string s = llvm::formatv("{0} {1}",
                                  "failed to parse struct pattern elements in "
                                  "parse struct pattern: ",
                                  pattern.getError())
                        .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
  }
  StructPatternElements el = pattern.getValue();
  pat.setElements(el);

  if (!check(lexer::TokenKind::BraceClose)) {
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse struct pattern");
  }
  assert(eat(TokenKind::BraceClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<StructPattern>(pat));
}

} // namespace rust_compiler::parser
