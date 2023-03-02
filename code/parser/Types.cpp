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

#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkPathExprSegment(uint8_t offset) {
  if (!checkPathIdentSegment(offset))
    return false;
  if (check(TokenKind::PathSep, offset + 1))
    return true;
  return true;
}

llvm::Expected<ast::GenericArgsConst> Parser::parseGenericArgsConst() {
  Location loc = getLocation();
  GenericArgsConst cons = {loc};

  if (check(TokenKind::BraceOpen)) {
    // block
    llvm::Expected<std::shared_ptr<ast::Expression>> block =
        parseBlockExpression();
    if (auto e = block.takeError()) {
      llvm::errs()
          << "failed to parse block expression in generic args const : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    cons.setBlock(*block);
  } else if (checkLiteral()) {
    // literal
    llvm::Expected<std::shared_ptr<ast::Expression>> literal =
        parseLiteralExpression();
    if (auto e = literal.takeError()) {
      llvm::errs()
          << "failed to parse literal expression in generic args const : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    cons.setLiteral(*literal);
  } else if (check(TokenKind::Minus) && checkLiteral(1)) {
    assert(eat(TokenKind::Minus));
    // minus literal
    cons.setLeadingMinus();
    llvm::Expected<std::shared_ptr<ast::Expression>> literal =
        parseLiteralExpression();
    if (auto e = literal.takeError()) {
      llvm::errs()
          << "failed to parse literal expression in generic args const : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    cons.setLiteral(*literal);
  } else if (checkSimplePathSegment()) {
    // simple path segment
    llvm::Expected<ast::SimplePathSegment> seg = parseSimplePathSegment();
    if (auto e = seg.takeError()) {
      llvm::errs()
          << "failed to parse simple path segment in generic args const : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    cons.setSegment(*seg);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse generic args const");
}

llvm::Expected<ast::GenericArgsBinding> Parser::parseGenericArgsBinding() {
  Location loc = getLocation();
  GenericArgsBinding binding = {loc};

  if (!check(TokenKind::Identifier)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse generic args binding: identifier");
  }
  binding.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (!check(TokenKind::Eq)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse generic args binding: =");
  }
  assert(eat(TokenKind::Eq));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type expression in generic args binding : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  binding.setType(*type);

  return binding;
}

llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
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
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse lifetime");
  }

  return std::make_shared<rust_compiler::ast::types::Lifetime>(lf);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplTraitType() {
  Location loc = getLocation();
  ImplTraitType trait = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IMPL))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse impl keyworkd in impl trait type");
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  llvm::Expected<ast::types::TypeParamBounds> bounds = parseTypeParamBounds();
  if (auto e = bounds.takeError()) {
    llvm::errs() << "failed to parse type param bounds in impl trait type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  trait.setBounds(*bounds);

  return std::make_shared<ImplTraitType>(trait);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseMacroInvocationType() {
  Location loc = getLocation();
  MacroInvocationType macro = {loc};

  llvm::Expected<ast::SimplePath> path = parseSimplePath();
  if (auto e = path.takeError()) {
    llvm::errs() << "failed to parse simple path in macro invocation type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  macro.setPath(*path);

  if (!check(TokenKind::Not))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse ! token in macro invocation type");
  assert(eat(TokenKind::Not));

  llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (auto e = token.takeError()) {
    llvm::errs()
        << "failed to parse delimt token tree in macro invocation type : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  macro.setTree(*token);

  return std::make_shared<MacroInvocationType>(macro);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectTypeOneBound() {
  Location loc = getLocation();

  TraitObjectTypeOneBound bound = {loc};

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    bound.setDyn();
    assert(eatKeyWord(KeyWordKind::KW_DYN));
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> traitBound =
      parseTraitBound();
  if (auto e = traitBound.takeError()) {
    llvm::errs()
        << "failed to parse trait bound in trait object type one bound : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bound.setBound(*traitBound);

  return std::make_shared<TraitObjectTypeOneBound>(bound);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseParenthesizedType() {
  Location loc = getLocation();
  ParenthesizedType parenType = {loc};

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in parenthesized type");
  assert(eat(TokenKind::ParenOpen));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in parenthesized type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  parenType.setType(*type);

  if (!check(TokenKind::ParenClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ) token in parenthesized type");
  assert(eat(TokenKind::ParenClose));

  return std::make_shared<ParenthesizedType>(parenType);
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTupleOrParensType() {
  CheckPoint cp = getCheckPoint();

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( token in tuple or parens type");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    recover(cp);
    return parseTupleType();
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in tuple or parens type : "
                 << toString(std::move(e)) << "\n";
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

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplTraitTypeOneBound() {
  Location loc = getLocation();
  ImplTraitTypeOneBound one = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse impl keyword in impl trait type one bound");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
      parseTraitBound();
  if (auto e = bound.takeError()) {
    llvm::errs()
        << "failed to parse trait bound in impl trait type one bound : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  one.setBound(*bound);

  return std::make_shared<ImplTraitTypeOneBound>(one);
}

llvm::Expected<ast::types::ForLifetimes> Parser::parseForLifetimes() {
  Location loc = getLocation();

  ForLifetimes forL = {loc};

  if (!checkKeyWord(KeyWordKind::KW_FOR)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse for keyword in for lifetimes");
  }
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
  if (auto e = genericParams.takeError()) {
    llvm::errs() << "failed to parse generic params in for lifetimes : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  forL.setGenericParams(*genericParams);

  return forL;
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplType() {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(inconvertibleErrorCode(),
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
      llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (auto e = bound.takeError()) {
        llvm::errs() << "failed to parse trait bound in impl type : "
                     << toString(std::move(e)) << "\n";
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

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
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
      llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (auto e = bound.takeError()) {
        llvm::errs() << "failed to parse trait bound in impl type : "
                     << toString(std::move(e)) << "\n";
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

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseArrayOrSliceType() {
  Location loc = getLocation();

  if (!check(TokenKind::SquareOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse array or slice type");
  }

  assert(eat(TokenKind::SquareOpen));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in array or slice type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::SquareClose)) {
    // slice type
    SliceType slice = {loc};
    slice.setType(*type);

    return std::make_shared<SliceType>(slice);
  } else if (check(TokenKind::Semi)) {
    // array type
    // slice type
    ArrayType arr = {loc};
    arr.setType(*type);

    assert(eat(TokenKind::Semi));

    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression in array type : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arr.setExpression(*expr);

    if (check(TokenKind::SquareClose)) {
      assert(eat(TokenKind::SquareClose));
      return std::make_shared<ArrayType>(arr);
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse array or slice type");
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseInferredType() {
  Location loc = getLocation();

  InferredType infer = {loc};

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    return std::make_shared<InferredType>(infer);
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse inferred type");
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseNeverType() {
  Location loc = getLocation();

  NeverType never = {loc};

  if (check(TokenKind::Not)) {
    assert(eat(TokenKind::Not));
    return std::make_shared<NeverType>(never);
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse never type");
}

llvm::Expected<ast::GenericArgs> Parser::parseGenericArgs() {
  Location loc = getLocation();
  GenericArgs args = {loc};

  if (!check(TokenKind::Lt)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse generic args");
  }
  assert(eat(TokenKind::Lt));

  if (check(TokenKind::Gt)) {
    assert(eat(TokenKind::Gt));
    return args;
  }

  std::optional<GenericArgKind> last = std::nullopt;

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse generic args: eof");
    } else if (check(TokenKind::Gt)) {
      assert(eat(TokenKind::Gt));
      return args;
    } else if (check(TokenKind::Comma) && check(TokenKind::Gt, 1)) {
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::Gt));
      args.setTrailingSemi();
      return args;
    } else {
      llvm::Expected<ast::GenericArg> arg = parseGenericArg(last);
      if (auto e = arg.takeError()) {
        llvm::errs() << "failed to parse generic arg in generic args : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      args.addArg(*arg);
      last = arg->getKind();
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse generic args");
}

llvm::Expected<ast::GenericArg>
Parser::parseGenericArg(std::optional<GenericArgKind> last) {
  Location loc = getLocation();
  GenericArg arg = {loc};

  if (checkLifetime()) {
    // lifetime
    llvm::Expected<ast::Lifetime> life = parseLifetimeAsLifetime();
    if (auto e = life.takeError()) {
      llvm::errs() << "failed to parse lifetime in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setLifetime(*life);
    return arg;
  } else if (checkIdentifier() && check(TokenKind::Eq, 1)) {
    // binding
    llvm::Expected<ast::GenericArgsBinding> bind = parseGenericArgsBinding();
    if (auto e = bind.takeError()) {
      llvm::errs() << "failed to parse generic args binding in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsBinding(*bind);
    return arg;
  } else if (checkLiteral()) {
    llvm::Expected<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (auto e = constArgs.takeError()) {
      llvm::errs() << "failed to parse generic args const in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(*constArgs);
    return arg;
  } else if (check(TokenKind::Minus) && checkLiteral(1)) {
    llvm::Expected<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (auto e = constArgs.takeError()) {
      llvm::errs() << "failed to parse generic args const in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(*constArgs);
    return arg;
  } else if (check(TokenKind::BraceOpen)) {
    llvm::Expected<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (auto e = constArgs.takeError()) {
      llvm::errs() << "failed to parse generic args const in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(*constArgs);
    return arg;
  } else if (checkSimplePathSegment() && check(TokenKind::Comma, 1)) {
    llvm::Expected<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (auto e = constArgs.takeError()) {
      llvm::errs() << "failed to parse generic args const in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(*constArgs);
    return arg;
  } else if (checkSimplePathSegment() && check(TokenKind::Lt, 1)) {
    llvm::Expected<ast::GenericArgsConst> constArgs = parseGenericArgsConst();
    if (auto e = constArgs.takeError()) {
      llvm::errs() << "failed to parse generic args const in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setArgsConst(*constArgs);
    return arg;
  } else {
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
        parseType();
    if (auto e = type.takeError()) {
      llvm::errs() << "failed to parse type expression in generic arg : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arg.setType(*type);
    return arg;
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse generic arg");
}

llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseTypeParamBound() {
  Location loc = getLocation();

  if (check(TokenKind::ParenOpen)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (check(TokenKind::QMark)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (check(TokenKind::PathSep)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (checkPathIdentSegment()) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> life =
      parseLifetimeAsTypeParamBound();
  if (auto e = life.takeError()) {
    llvm::errs() << "failed to parse lifetime in type param bound : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  return *life;
}

llvm::Expected<ast::types::TypeParamBounds> Parser::parseTypeParamBounds() {
  Location loc = getLocation();

  TypeParamBounds bounds = {loc};

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> first =
      parseTypeParamBound();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse  type param bound in type param bounds :"
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bounds.addTypeParamBound(*first);

  // heuristic ; or , or { or >
  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse type param bounds: eof");
    } else if (check(TokenKind::Semi) || check(TokenKind::Comma) ||
               check(TokenKind::BraceOpen) || check(TokenKind::Lt)) {
      return bounds;
    } else if (check(TokenKind::Plus) &&
               (check(TokenKind::Semi, 1) || check(TokenKind::Comma, 1) ||
                check(TokenKind::BraceOpen, 1) || check(TokenKind::Lt, 1))) {
      bounds.setTrailingPlus();
      return bounds;
    } else if (check(TokenKind::Plus)) {
      assert(eat(TokenKind::Plus));
      llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> type =
          parseTypeParamBound();
      if (auto e = first.takeError()) {
        llvm::errs()
            << "failed to parse  type param bound in type param bounds :"
            << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      bounds.addTypeParamBound(*type);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse type param bounds");
    }
  }
  return createStringError(inconvertibleErrorCode(),
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
    } else if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
      assert(eatKeyWord(KeyWordKind::KW_DOLLARCRATE));
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

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseType() {

  llvm::outs() << "parseType"
               << "\n";

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

llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
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
      llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (auto e = forL.takeError()) {
        llvm::errs() << "failed to parse ForLifetimes in trait bound : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(*forL);
    }
    llvm::Expected<std::shared_ptr<ast::types::TypePath>> path =
        parseTypePath();
    if (auto e = path.takeError()) {
      llvm::errs() << "failed to parse type path in trait bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(*path);
    if (!check(TokenKind::ParenClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse trait bound ;missing ): ");
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<TraitBound>(tr);
  } else if (!check(TokenKind::ParenOpen)) {
    if (check(TokenKind::QMark)) {
      assert(eat(TokenKind::QMark));
      tr.setHasQuestionMark();
    }

    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (auto e = forL.takeError()) {
        llvm::errs() << "failed to parse ForLifetimes in trait bound : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(*forL);
    }

    llvm::Expected<std::shared_ptr<ast::types::TypePath>> path =
        parseTypePath();
    if (auto e = path.takeError()) {
      llvm::errs() << "failed to parse type path in trait bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(*path);
    return std::make_shared<TraitBound>(tr);
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse trait bound: ");
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse trait bound: ");
}

} // namespace rust_compiler::parser
