#include "AST/GenericArg.h"
#include "AST/GenericArgs.h"
#include "AST/GenericArgsBinding.h"
#include "AST/GenericArgsConst.h"
#include "AST/Lifetime.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/ImplTraitType.h"
#include "AST/Types/ImplTraitTypeOneBound.h"
#include "AST/Types/InferredType.h"
#include "AST/Types/Lifetime.h"
#include "AST/Types/MacroInvocationType.h"
#include "AST/Types/NeverType.h"
#include "AST/Types/ParenthesizedType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TraitObjectTypeOneBound.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypePath.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkPathExprSegment(uint8_t offset) {
  if (!checkPathIdentSegment(offset))
    return false;
  if (check(TokenKind::PathSep, offset + 1))
    return true;
  return true;
}

// Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
// Parser::parseArrayOrSliceType() {
//   CheckPoint cp = getCheckPoint();
//
//   if (!check(TokenKind::SquareOpen)) {
//     // report error
//     return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
//         "failed to parse array or slice type: missed [ token");
//   }
//   assert(eat(TokenKind::SquareOpen));
//
//   Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
//       parseType();
//   if (!type) {
//     // report error
//     llvm::errs() << "failed to parse type in array or slice type: "
//                  << type.getError() << "\n";
//     std::string s =
//         llvm::formatv("{0}\n{1}", "failed to parse type in array or slice
//         type",
//                       type.getError())
//             .str();
//     return Result<std::shared_ptr<ast::types::TypeExpression>,
//     std::string>(s);
//   }
//
//   if (check(TokenKind::Semi)) {
//     recover(cp);
//     return parseArrayType();
//   } else if (check(TokenKind::SquareClose)) {
//     recover(cp);
//     return parseSliceType();
//   }
//   llvm::errs() << "failed to parse array or slice type"
//                << "\n";
//       return Result<std::shared_ptr<ast::types::TypeExpression>,
//       std::string>(
//           "failed to parse array or slice type");
// }

Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
Parser::parseSliceType() {
  SliceType slice = {getLocation()};

  llvm::errs() << "parseSliceType"
               << "\n";

  if (!check(TokenKind::SquareOpen)) {
    // report error
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
        "failed to parse slice type: missed [ token");
  }
  assert(eat(TokenKind::SquareOpen));

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    // report error
    llvm::errs() << "failed to parse type in slice type: " << type.getError()
                 << "\n";
    std::string s =
        llvm::formatv("{0}\n{1}", "failed to parse type in slice type",
                      type.getError())
            .str();
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(s);
  }
  slice.setType(type.getValue());

  if (!check(TokenKind::SquareClose)) {
    // report error
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
        "failed to parse slice type: missed ] token");
  }
  assert(eat(TokenKind::SquareClose));

  return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
      std::make_shared<SliceType>(slice));
}

Result<std::shared_ptr<ast::types::TypeExpression>, std::string>
Parser::parseArrayType() {
  ArrayType ar = {getLocation()};

  if (!check(TokenKind::SquareOpen)) {
    // report error
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
        "failed to parse slice type: missed [ token");
  }
  assert(eat(TokenKind::SquareOpen));

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    // report error
    llvm::errs() << "failed to parse type in array type: " << type.getError()
                 << "\n";
    std::string s =
        llvm::formatv("{0}\n{1}", "failed to parse type in array type",
                      type.getError())
            .str();
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(s);
  }
  ar.setType(type.getValue());

  if (!check(TokenKind::Semi)) {
    // report error
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
        "failed to parse array type: missed ; token");
  }
  assert(eat(TokenKind::Semi));

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    // report error
    llvm::errs() << "failed to parse expression in array type: "
                 << expr.getError() << "\n";
    std::string s =
        llvm::formatv("{0}\n{1}", "failed to parse expression in array type",
                      expr.getError())
            .str();
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(s);
  }
  ar.setExpression(expr.getValue());

  if (!check(TokenKind::SquareClose)) {
    // report error
    return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
        "failed to parse array type: missed ] token");
  }
  assert(eat(TokenKind::SquareClose));

  return Result<std::shared_ptr<ast::types::TypeExpression>, std::string>(
      std::make_shared<ArrayType>(ar));
}

StringResult<ast::GenericArgsConst> Parser::parseGenericArgsConst() {
  Location loc = getLocation();
  GenericArgsConst cons = {loc};

  if (check(TokenKind::BraceOpen)) {
    // block
    StringResult<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression({});
    if (!block) {
      llvm::errs() << "failed to parse block expression in generic args const: "
                   << block.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    cons.setBlock(block.getValue());
    return StringResult<ast::GenericArgsConst>(cons);
  } else if (checkLiteral()) {
    // literal
    StringResult<std::shared_ptr<ast::Expression>> literal =
        parseLiteralExpression({});
    if (!literal) {
      llvm::errs()
          << "failed to parse literal expression in generic args const: "
          << literal.getError() << "\n";
      // printFunctionStack();
      return StringResult<ast::GenericArgsConst>(
          llvm::formatv(
              "{} {}",
              "failed to parse literal expression in generic args const: ",
              literal.getError())
              .str());
    }
    cons.setLiteral(literal.getValue());
    return StringResult<ast::GenericArgsConst>(cons);
  } else if (check(TokenKind::Minus) && checkLiteral(1)) {
    assert(eat(TokenKind::Minus));
    // minus literal
    cons.setLeadingMinus();
    StringResult<std::shared_ptr<ast::Expression>> literal =
        parseLiteralExpression({});
    if (!literal) {
      llvm::errs()
          << "failed to parse literal expression in generic args const: "
          << literal.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    cons.setLiteral(literal.getValue());
    return StringResult<ast::GenericArgsConst>(cons);
  } else if (checkSimplePathSegment()) {
    // simple path segment
    StringResult<ast::SimplePathSegment> seg = parseSimplePathSegment();
    if (!seg) {
      llvm::errs()
          << "failed to parse simple path segment in generic args const: "
          << seg.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    cons.setSegment(seg.getValue());
    return StringResult<ast::GenericArgsConst>(cons);
  }
  return StringResult<ast::GenericArgsConst>(
      "failed to parse generic args const");
}

StringResult<ast::GenericArgsBinding> Parser::parseGenericArgsBinding() {
  Location loc = getLocation();
  GenericArgsBinding binding = {loc};

  if (!check(TokenKind::Identifier)) {
    return StringResult<ast::GenericArgsBinding>(
        "failed to parse generic args binding: identifier");
  }
  binding.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Eq)) {
    return StringResult<ast::GenericArgsBinding>(
        "failed to parse generic args binding: =");
  }
  assert(eat(TokenKind::Eq));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in generic args binding: "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  binding.setType(type.getValue());

  return StringResult<ast::GenericArgsBinding>(binding);
}

StringResult<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseLifetimeAsTypeParamBound() {
  Location loc = getLocation();
  rust_compiler::ast::types::Lifetime lf = {loc};

  if (check(TokenKind::LIFETIME_OR_LABEL)) {
    lf.setLifetime(getToken().getStorage());
  } else if (checkKeyWord(KeyWordKind::KW_STATICLIFETIME)) {
    lf.setLifetime(getToken().getStorage());
  } else if (check(TokenKind::LIFETIME_TOKEN) &&
             getToken().getStorage() == "'_") {
    lf.setLifetime(getToken().getStorage());
  } else {
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        "failed to parse lifetime");
  }

  return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
      std::make_shared<rust_compiler::ast::types::Lifetime>(lf));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplTraitType() {
  Location loc = getLocation();
  ImplTraitType trait = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IMPL))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse impl keyworkd in impl trait type");
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  StringResult<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
  if (!bounds) {
    llvm::errs() << "failed to parse type param bounds in impl trait type: "
                 << bounds.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  trait.setBounds(bounds.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<ImplTraitType>(trait));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseMacroInvocationType() {
  Location loc = getLocation();
  MacroInvocationType macro = {loc};

  StringResult<ast::SimplePath> path = parseSimplePath();
  if (!path) {
    llvm::errs() << "failed to parse simple path in macro invocation type: "
                 << path.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setPath(path.getValue());

  if (!check(TokenKind::Not))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ! token in macro invocation type");
  assert(eat(TokenKind::Not));

  StringResult<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (!token) {
    llvm::errs()
        << "failed to parse delim token tree in macro invocation type: "
        << token.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  macro.setTree(token.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<MacroInvocationType>(macro));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectTypeOneBound() {
  Location loc = getLocation();

  TraitObjectTypeOneBound bound = {loc};

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    bound.setDyn();
    assert(eatKeyWord(KeyWordKind::KW_DYN));
  }

  StringResult<std::shared_ptr<ast::types::TypeParamBound>> traitBound =
      parseTraitBound();
  if (!traitBound) {
    llvm::errs()
        << "failed to parse trait bound in trait object type one bound: "
        << traitBound.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  bound.setBound(traitBound.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<TraitObjectTypeOneBound>(bound));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseParenthesizedType() {
  Location loc = getLocation();
  ParenthesizedType parenType = {loc};

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ( token in parenthesized type");
  assert(eat(TokenKind::ParenOpen));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in parenthesized type: "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  parenType.setType(type.getValue());

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ) token in parenthesized type");
  assert(eat(TokenKind::ParenClose));

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<ParenthesizedType>(parenType));
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTupleOrParensType() {
  CheckPoint cp = getCheckPoint();

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ( token in tuple or parens type");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    recover(cp);
    return parseTupleType();
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in tuple or parenthesized type: "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::ParenClose)) {
    // assert(eat(TokenKind::ParenClose));
    recover(cp);
    return parseParenthesizedType();
  }

  recover(cp);
  return parseTupleType();
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplTraitTypeOneBound() {
  Location loc = getLocation();
  ImplTraitTypeOneBound one = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse impl keyword in impl trait type one bound");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  StringResult<std::shared_ptr<ast::types::TypeParamBound>> bound =
      parseTraitBound();
  if (!bound) {
    llvm::errs() << "failed to parse trait bound in impl trait type one bound: "
                 << bound.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  one.setBound(bound.getValue());

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      std::make_shared<ImplTraitTypeOneBound>(one));
}

StringResult<ast::types::ForLifetimes> Parser::parseForLifetimes() {
  Location loc = getLocation();

  ForLifetimes forL = {loc};

  if (!checkKeyWord(KeyWordKind::KW_FOR)) {
    return StringResult<ast::types::ForLifetimes>(
        "failed to parse for keyword in for lifetimes");
  }
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  StringResult<ast::GenericParams> genericParams = parseGenericParams();
  if (!genericParams) {
    llvm::errs() << "failed to parse generic params in for lifetimes: "
                 << genericParams.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  forL.setGenericParams(genericParams.getValue());

  return StringResult<ast::types::ForLifetimes>(forL);
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplType() {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse impl keyword in impl trait");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  while (true) {
    if (check(TokenKind::LIFETIME_OR_LABEL)) {
      // TypeParamBounds
      recover(cp);
      return parseImplTraitType();
    } else if (check(TokenKind::LIFETIME_TOKEN) &&
               checkStaticOrUnderscore()) { // 'static or '_ FIXME
      // TypeParamBound
      recover(cp);
      return parseImplTraitType();
    } else if (check(TokenKind::QMark) || checkKeyWord(KeyWordKind::KW_FOR) ||
               check(TokenKind::PathSep) || checkPathIdentSegment()) {
      // TraitBound
      StringResult<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (!bound) {
        llvm::errs() << "failed to parse trait bound in impl type: "
                     << bound.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::Plus)) {
        recover(cp);
        return parseImplTraitType();
      } else {
        recover(cp);
        return parseImplTraitTypeOneBound();
      }
    }
  }
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectType() {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    assert(eatKeyWord(KeyWordKind::KW_DYN));
  }

  while (true) {
    if (check(TokenKind::LIFETIME_OR_LABEL)) {
      // TypeParamBounds
      recover(cp);
      return parseTraitObjectType();
    } else if (check(TokenKind::LIFETIME_TOKEN) &&
               checkStaticOrUnderscore()) { // 'static or '_ FIXME
      // TypeParamBound
      recover(cp);
      return parseTraitObjectType();
    } else if (check(TokenKind::QMark) || checkKeyWord(KeyWordKind::KW_FOR) ||
               check(TokenKind::PathSep) || checkPathIdentSegment()) {
      // TraitBound
      StringResult<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (!bound) {
        llvm::errs() << "failed to parse trait bound in trait object type: "
                     << bound.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::Plus)) {
        recover(cp);
        return parseTraitObjectType();
      } else {
        recover(cp);
        return parseTraitObjectTypeOneBound();
      }
    }
  }
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseArrayOrSliceType() {
  Location loc = getLocation();

  llvm::errs() << "parseArrayOrSliceType"
               << "\n";

  if (!check(TokenKind::SquareOpen)) {
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse array or slice type");
  }

  assert(eat(TokenKind::SquareOpen));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in array or slice type: "
                 << type.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0}\n{1}",
            "failed to parse type in array or slice type: ", type.getError())
            .str();
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
  }

  if (check(TokenKind::SquareClose)) {
    assert(eat(TokenKind::SquareClose));
    // slice type
    SliceType slice = {loc};
    slice.setType(type.getValue());

    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<SliceType>(slice));
  } else if (check(TokenKind::Semi)) {
    // array type
    ArrayType arr = {loc};
    arr.setType(type.getValue());

    if (!check(TokenKind::Semi)) {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          "failed to parse ; token in array or slice type");
    }
    assert(eat(TokenKind::Semi));

    Restrictions restrictions;
    StringResult<std::shared_ptr<ast::Expression>> expr =
        parseExpression({}, restrictions);
    if (!expr) {
      llvm::errs() << "failed to parse expression in array or slice type: "
                   << expr.getError() << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv("{0}\n{1}",
                        "failed to parse expression in array or slice type: ",
                        expr.getError())
              .str();
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(s);
    }
    arr.setExpression(expr.getValue());

    if (!check(TokenKind::SquareClose)) {
      return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
          "failed to parse ] token in array or slice type");
    }
    assert(eat(TokenKind::SquareClose));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<ArrayType>(arr));
  }

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      "failed to parse array or slice type");
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseInferredType() {
  Location loc = getLocation();

  InferredType infer = {loc};

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<InferredType>(infer));
  }

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      "failed to parse inferred type");
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseNeverType() {
  Location loc = getLocation();

  NeverType never = {loc};

  if (check(TokenKind::Not)) {
    assert(eat(TokenKind::Not));
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<NeverType>(never));
  }

  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      "failed to parse never type");
}

StringResult<ast::GenericArgs> Parser::parseGenericArgs() {
  Location loc = getLocation();
  GenericArgs args = {loc};

  if (!check(TokenKind::Lt)) {
    return StringResult<ast::GenericArgs>(
        "failed to parse < token in generic args");
  }
  assert(eat(TokenKind::Lt));

  if (check(TokenKind::Gt)) {
    assert(eat(TokenKind::Gt));
    return StringResult<ast::GenericArgs>(args);
  }

  std::optional<GenericArgKind> last = std::nullopt;

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::GenericArgs>(
          "failed to parse generic args: eof");
    } else if (check(TokenKind::Gt)) {
      assert(eat(TokenKind::Gt));
      return StringResult<ast::GenericArgs>(args);
    } else if (check(TokenKind::Comma) && check(TokenKind::Gt, 1)) {
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::Gt));
      args.setTrailingSemi();
      return StringResult<ast::GenericArgs>(args);
    } else {
      StringResult<ast::GenericArg> arg = parseGenericArg(last);
      if (!arg) {
        llvm::errs() << "failed to parse generic arg in generic args: "
                     << arg.getError() << "\n";
        printFunctionStack();
        // exit(EXIT_FAILURE);
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse generic arg in generic args: ", arg.getError())
                .str();
        return StringResult<ast::GenericArgs>(s);
      }
      args.addArg(arg.getValue());
      last = arg.getValue().getKind();
    }
  }
  return StringResult<ast::GenericArgs>("failed to parse generic args");
}

StringResult<ast::GenericArg>
Parser::parseGenericArg(std::optional<GenericArgKind> last) {
  Location loc = getLocation();
  GenericArg arg = {loc};

  llvm::errs() << "parseGenericArg"
               << "\n";

  if (checkLifetime()) {
    // lifetime
    StringResult<ast::Lifetime> life = parseLifetimeAsLifetime();
    if (!life) {
      llvm::errs() << "failed to parse lifetime as lifetime in generic arg: "
                   << life.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setLifetime(life.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (checkIdentifier() && check(TokenKind::Eq, 1)) {
    // binding
    StringResult<ast::GenericArgsBinding> bind = parseGenericArgsBinding();
    if (!bind) {
      llvm::errs() << "failed to parse generic args binding in generic arg: "
                   << bind.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsBinding(bind.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (checkLiteral()) {
    StringResult<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (!constArgs) {
      llvm::errs() << "failed to parse generic args const in generic arg: "
                   << constArgs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(constArgs.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (check(TokenKind::Minus) && checkLiteral(1)) {
    StringResult<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (!constArgs) {
      llvm::errs() << "failed to parse generic args const in generic arg: "
                   << constArgs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(constArgs.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (check(TokenKind::BraceOpen)) {
    StringResult<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (!constArgs) {
      llvm::errs() << "failed to parse generic args const in generic arg: "
                   << constArgs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(constArgs.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (canTokenStartType(getToken())) {
    StringResult<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (!type) {
      llvm::errs() << "failed to parse type in generic arg: " << type.getError()
                   << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}", "failed to parse type in generic arg: ",
                        type.getError())
              .str();
      // exit(EXIT_FAILURE);
      return StringResult<ast::GenericArg>(s);
    }
    arg.setType(type.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (checkSimplePathSegment() && check(TokenKind::Comma, 1)) {
    StringResult<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (!constArgs) {
      llvm::errs() << "failed to parse generic args const in generic arg: "
                   << constArgs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(constArgs.getValue());
    return StringResult<ast::GenericArg>(arg);
  } else if (checkSimplePathSegment() && check(TokenKind::Lt, 1)) {
    StringResult<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (!constArgs) {
      llvm::errs() << "failed to parse generic args const in generic arg: "
                   << constArgs.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(constArgs.getValue());
    return StringResult<ast::GenericArg>(arg);
  }
  return StringResult<ast::GenericArg>("failed to parse generic arg");
}

StringResult<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseTypeParamBound() {
  Location loc = getLocation();

  if (check(TokenKind::ParenOpen)) {
    StringResult<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (!trait) {
      llvm::errs() << "failed to parse trait bound in type param bound: "
                   << trait.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        trait.getValue());
  } else if (check(TokenKind::QMark)) {
    StringResult<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (!trait) {
      llvm::errs() << "failed to parse trait bound in type param bound: "
                   << trait.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        trait.getValue());
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    StringResult<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (!trait) {
      llvm::errs() << "failed to parse trait bound in type param bound: "
                   << trait.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        trait.getValue());
  } else if (check(TokenKind::PathSep)) {
    StringResult<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (!trait) {
      llvm::errs() << "failed to parse trait bound in type param bound: "
                   << trait.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        trait.getValue());
  } else if (checkPathIdentSegment()) {
    StringResult<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (!trait) {
      llvm::errs() << "failed to parse trait bound in type param bound: "
                   << trait.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        trait.getValue());
  }

  StringResult<std::shared_ptr<ast::types::TypeParamBound>> life =
      parseLifetimeAsTypeParamBound();
  if (!life) {
    llvm::errs()
        << "failed to parse lifetime as type param bound in type param bound: "
        << life.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
      life.getValue());
}

StringResult<ast::types::TypeParamBounds> Parser::parseTypeParamBounds() {
  Location loc = getLocation();

  TypeParamBounds bounds = {loc};

  StringResult<std::shared_ptr<ast::types::TypeParamBound>> first =
      parseTypeParamBound();
  if (!first) {
    llvm::errs() << "failed to parse type param bound in type param bound: "
                 << first.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  bounds.addTypeParamBound(first.getValue());

  // heuristic ; or , or { or >
  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<ast::types::TypeParamBounds>(
          "failed to parse type param bounds: eof");
    } else if (check(TokenKind::Semi) || check(TokenKind::Comma) ||
               check(TokenKind::BraceOpen) || check(TokenKind::Lt)) {
      return StringResult<ast::types::TypeParamBounds>(bounds);
    } else if (check(TokenKind::Plus) &&
               (check(TokenKind::Semi, 1) || check(TokenKind::Comma, 1) ||
                check(TokenKind::BraceOpen, 1) || check(TokenKind::Lt, 1))) {
      bounds.setTrailingPlus();
      return StringResult<ast::types::TypeParamBounds>(bounds);
    } else if (check(TokenKind::Plus)) {
      assert(eat(TokenKind::Plus));
      StringResult<std::shared_ptr<ast::types::TypeParamBound>> type =
          parseTypeParamBound();
      if (!type) {
        llvm::errs() << "failed to parse type param bound in type param bound: "
                     << type.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      bounds.addTypeParamBound(type.getValue());
    } else {
      return StringResult<ast::types::TypeParamBounds>(
          "failed to parse type param bounds");
    }
  }
  return StringResult<ast::types::TypeParamBounds>(
      "failed to parse type param bounds");
}

PathKind Parser::testTypePathOrSimplePath() {
  if (check(TokenKind::PathSep)) {
    assert(eat(TokenKind::PathSep));
  }

  while (true) {
    if (check(TokenKind::Identifier)) {
      assert(eat(TokenKind::Identifier));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
      assert(eatKeyWord(KeyWordKind::KW_SUPER));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
      assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
      assert(eatKeyWord(KeyWordKind::KW_CRATE));
      continue;
    } else if (check(TokenKind::Dollar) &&
               checkKeyWord(KeyWordKind::KW_CRATE)) {
      assert(eat(TokenKind::Dollar));
      assert(eatKeyWord(KeyWordKind::KW_CRATE));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SELFTYPE)) {
      return PathKind::TypePath;
    } else if (check(TokenKind::Lt)) {
      // GenericArgs
      return PathKind::TypePath;
    } else if (check(TokenKind::ParenOpen)) {
      // TypePathFn
      return PathKind::TypePath;
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      continue;
    } else if (check(TokenKind::Eof)) {
      return PathKind::Unknown;
    } else {
      return PathKind::SimplePath;
    }
  }
  // SimplePath?
}
// TypePath: PathIdentSegment
// SimplePath: SimplePathSegment

// eof

// llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
//     parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType()
//     {
//   CheckPoint qp = getCheckPoint();
//
//   if (checkKeyWord(KeyWordKind::KW_FOR)) {
//     llvm::Expected<ForLifetimes> forLifetimes = parseForLifetimes();
//     if (auto e = forLifetimes.takeError()) {
//       llvm::errs() << "failed to parse for lifetime in ... : "
//                    << toString(std::move(e)) << "\n";
//       exit(EXIT_FAILURE);
//     }
//
//     if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_FN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (check(TokenKind::PathSep)) {
//       // TraitObjectType
//       // TypePath
//       // MacroInvocation
//     } else if (check(TokenKind::)) {
//     }
//   } else { // not for
//     if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_FN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (check(TokenKind::PathSep)) {
//       // TraitObjectType: TypePath
//       // TypePath: TypePath
//       // MacroInvocation: SimplePath
//     } else if () {
//     }
//     // BareFunctionType or TraitObjectType
//   }
//   // else if (check(TokenKind::ParenOpen)) {
//   //   // ParenType or TupleType or TraitObjectType
//   //   //    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
//   types
//   //   =
//   //   //       parseType();
//   // }
//   // else if (check(TokenKind::PathSeq)) {
//   //   // SimplePath in MacroInvocation
//   //   // TypePath
//   //   // Or TraitObjectType
//   // }
// }

StringResult<std::shared_ptr<ast::types::TypeExpression>> Parser::parseType() {

  //  llvm::errs() << "parseType: " << Token2String(getToken().getKind()) <<
  //  "\n";

  if (checkKeyWord(KeyWordKind::KW_IMPL))
    return parseImplType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

  if (check(TokenKind::Star))
    return parseRawPointerType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  if (check(TokenKind::Not))
    return parseNeverType();

  if (check(TokenKind::And))
    return parseReferenceType();

  if (check(TokenKind::Underscore))
    return parseInferredType();

  if (check(TokenKind::Lt))
    return parseQualifiedPathInType();

  if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1))
    return parseTupleType();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_EXTERN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FN))
    return parseBareFunctionType();

  return parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType();
}

StringResult<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseTraitBound() {
  Location loc = getLocation();
  TraitBound tr = {loc};

  if (check(TokenKind::ParenOpen)) {
    tr.setHasParenthesis();
    assert(eat(TokenKind::ParenOpen));
    if (check(TokenKind::QMark)) {
      assert(eat(TokenKind::QMark));
      tr.setHasQuestionMark();
    }
    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      StringResult<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (!forL) {
        llvm::errs() << "failed to parse for life times in trait bound: "
                     << forL.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(forL.getValue());
    }
    StringResult<std::shared_ptr<ast::types::TypeExpression>> path =
        parseTypePath();
    if (!path) {
      llvm::errs() << "failed to parse type path in trait bound: "
                   << path.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(path.getValue());
    if (!check(TokenKind::ParenClose))
      return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
          "failed to parse trait bound ;missing ): ");
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        std::make_shared<TraitBound>(tr));
  } else if (!check(TokenKind::ParenOpen)) {
    if (check(TokenKind::QMark)) {
      assert(eat(TokenKind::QMark));
      tr.setHasQuestionMark();
    }

    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      StringResult<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (!forL) {
        llvm::errs() << "failed to parse for life times in trait bound: "
                     << forL.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(forL.getValue());
    }

    StringResult<std::shared_ptr<ast::types::TypeExpression>> path =
        parseTypePath();
    if (!path) {
      llvm::errs() << "failed to parse type path in trait bound: "
                   << path.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(path.getValue());
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        std::make_shared<TraitBound>(tr));
  } else {
    return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
        "failed to parse trait bound: ");
  }

  return StringResult<std::shared_ptr<ast::types::TypeParamBound>>(
      "failed to parse trait bound: ");
}

} // namespace rust_compiler::parser
