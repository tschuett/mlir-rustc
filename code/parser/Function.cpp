#include "Function.h"

#include "AST/Function.h"
#include "AST/FunctionQualifiers.h"
#include "AST/GenericParams.h"
#include "AST/Type.h"
#include "AST/WhereClause.h"
#include "Generics.h"
#include "Lexer/Token.h"
#include "WhereClause.h"

#include <llvm/Support/raw_os_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<ast::FunctionQualifiers>
tryParseFunctionQualifiers(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;
  FunctionQualifiers qual{view.front().getLocation()};

  llvm::outs() << "tryParseFunctionQualifiers: start" << "\n";

  if (view.front().getKind() == TokenKind::Keyword) {
    if (view.front().getIdentifier() == "const") {
      qual.setConst();
      view = view.subspan(1);
    }
    if (view.front().getIdentifier() == "async") {
      qual.setAsync();
      view = view.subspan(1);
    }
    if (view.front().getIdentifier() == "unsafe") {
      qual.setUnsafe();
      view = view.subspan(1);
    }
    if (view.front().getIdentifier() == "extern") {
      // qual.setExtern();
      // view = view.subspan(1);
      // FIXME Abi
    }
  }
  return qual;
}

std::optional<std::shared_ptr<ast::types::Type>>
tryParseFunctionReturnType(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().getKind() != TokenKind::ThinArrow)
    return std::nullopt;

  view = view.subspan(1);

  std::optional<std::shared_ptr<ast::types::Type>> type = tryParseType(view);

  return type;
}

std::optional<ast::FunctionSignature>
tryParseFunctionSignature(std::span<lexer::Token> tokens) {
  std::span<Token> view = tokens;
  FunctionSignature sig{view.front().getLocation()};

  printf("tryParseFunctionSignature: start\n");

  std::optional<ast::FunctionQualifiers> qual =
      tryParseFunctionQualifiers(view);

  if (!qual) {
    // FIXME
  }

  printf("qualifiers found %zu\n", (*qual).getTokens());
  sig.setQualifiers(*qual);
  view = view.subspan((*qual).getTokens());

  if (view.front().getKind() == TokenKind::Keyword &&
      view.front().getIdentifier() == "fn") {
    view = view.subspan(1);
  } else {
    printTokenState(view);
    printf("tryParseFunctionSignature: found no fn\n");
    return std::nullopt;
  }

  if (view.front().getKind() == TokenKind::Identifier) {
    sig.setName(view.front().getIdentifier());
    view = view.subspan(1);
  } else {
    printf("tryParseFunctionSignature: found no name\n");
    return std::nullopt;
  }

  std::optional<std::shared_ptr<GenericParams>> generic =
      tryParseGenericParams(view);

  if (generic) {
    sig.setGenericParams(*generic);
  }

  if (view.front().getKind() == TokenKind::ParenOpen) {
    view = view.subspan(1);
  } else {
    printf("tryParseFunctionSignature: found no (\n");
    return std::nullopt;
  }

  std::optional<FunctionParameters> params = tryParseFunctionParameters(view);

  if (params) {
    sig.setParameters(*params);
    view = view.subspan((*params).getTokens());
  } else {
    // FIXME
    printf("tryParseFunctionSignature: found no parameters\n");
    //return std::nullopt;
  }

  if (view.front().getKind() == TokenKind::ParenClose) {
    view = view.subspan(1);
  } else {
    printf("tryParseFunctionSignature: found no )\n");
    return std::nullopt;
  }

  std::optional<std::shared_ptr<ast::types::Type>> returnType =
      tryParseFunctionReturnType(view);

  if (returnType) {
    sig.setReturnType(*returnType);
  }

  std::optional<WhereClause> where = tryParseWhereClause(view);
  if (where) {
    sig.setWhereClause(*where);
  }

  return sig;
}

std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                              std::string_view modulePath) {
  std::span<Token> view = tokens;

  Function f = {view.front().getLocation()};

  llvm::outs() << "tryParseFunction: start" << "\n";
  //printTokenState(view);

  std::optional<FunctionSignature> sig = tryParseFunctionSignature(view);

  if (sig) {
    f.setSignature(*sig);
  } else {
    printf("tryParseFunction: found no signature\n");
    return std::nullopt;
  }

  if (view.front().getKind() == TokenKind::Semi) {
    return f;
  }

  std::optional<std::shared_ptr<BlockExpression>> block =
      tryParseBlockExpression(view);
  if (block) {
    f.setBody(*block);
    return f;
  }

  printf("tryParseFunction: found function\n");

  return f;
  return std::nullopt;
}

} // namespace rust_compiler::parser

/*
  TODO:

  https://doc.rust-lang.org/reference/items/generics.html
 */
