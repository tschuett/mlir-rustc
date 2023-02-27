#include "AST/MacroFragSpec.h"
#include "AST/MacroMatcher.h"
#include "AST/MacroRepOp.h"
#include "AST/MacroRepSep.h"
#include "AST/MacroRule.h"
#include "AST/MacroRulesDef.h"
#include "AST/MacroRulesDefinition.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/ADT/StringSwitch.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::MacroMatch> Parser::parseMacroMatch() {
  Location loc = getLocation();
  MacroMatch match = {loc};
}

llvm::Expected<ast::MacroRepSep> Parser::parseMacroRepSep() {
  Location loc = getLocation();

  MacroRepSep sep = {loc};

  if (checkDelimiters()) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse macro rep sep");
  } else if (check(TokenKind::Star)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse macro rep sep");
  } else if (check(TokenKind::Plus)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse macro rep sep");
  } else if (check(TokenKind::QMark)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse macro rep sep");
  }

  sep.setToken(getToken());
  return sep;
}

llvm::Expected<ast::MacroRepOp> Parser::parseMacroRepOp() {
  Location loc = getLocation();
  MacroRepOp op = {loc};

  if (check(TokenKind::Star)) {
    op.setKind(MacroRepOpKind::Star);
  } else if (check(TokenKind::Plus)) {
    op.setKind(MacroRepOpKind::Plus);
  } else if (check(TokenKind::QMark)) {
    op.setKind(MacroRepOpKind::Qmark);
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse marco rep op");
  }

  return op;
}

llvm::Expected<ast::MacroFragSpec> Parser::parseMacroFragSpec() {
  Location loc = getLocation();
  MacroFragSpec frag = {loc};

  if (!checkIdentifier())
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse marco frag spec");

  MacroFragSpecKind k =
      StringSwitch<MacroFragSpecKind>(getToken().getIdentifier())
          .Case("block", MacroFragSpecKind::Block)
          .Case("expr", MacroFragSpecKind::Expr)
          .Case("ident", MacroFragSpecKind::Ident)
          .Case("item", MacroFragSpecKind::Item)
          .Case("lifetime", MacroFragSpecKind::Lifetime)
          .Case("literal", MacroFragSpecKind::Literal)
          .Case("meta", MacroFragSpecKind::Meta)
          .Case("pat", MacroFragSpecKind::Pat)
          .Case("pat_param", MacroFragSpecKind::PatParam)
          .Case("path", MacroFragSpecKind::Path)
          .Case("stmt", MacroFragSpecKind::Stmt)
          .Case("tt", MacroFragSpecKind::Tt)
          .Case("ty", MacroFragSpecKind::Ty)
          .Case("vis", MacroFragSpecKind::Vis)
          .Default(MacroFragSpecKind::Unknown);

  frag.setKind(k);

  return frag;
}

llvm::Expected<ast::MacroTranscriber> Parser::parseMacroTranscriber() {
  Location loc = getLocation();
  MacroTranscriber transcriber = {loc};

  llvm::Expected<std::shared_ptr<ast::DelimTokenTree>> tree =
      parseDelimTokenTree();
  if (auto e = tree.takeError()) {
    llvm::errs() << "failed to parse delm token tree "
                    "in macro transcriber : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  transcriber.setTree(*tree);

  return transcriber;
}

llvm::Expected<ast::MacroMatcher> Parser::parseMacroMatcher() {
  Location loc = getLocation();
  MacroMatcher matcher = {loc};

  if (check(TokenKind::ParenOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse marco matcher: eof");
      } else if (check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::ParenClose));
        matcher.setKind(MacroMatcherKind::Paren);
        return matcher;
      } else {
        llvm::Expected<ast::MacroMatch> match = parseMacroMatch();
        if (auto e = match.takeError()) {
          llvm::errs() << "failed to parse macro match in "
                          "macro matcher : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(*match);
      }
    }
  } else if (check(TokenKind::SquareOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse marco matcher: eof");
      } else if (check(TokenKind::SquareClose)) {
        assert(eat(TokenKind::SquareClose));
        matcher.setKind(MacroMatcherKind::Square);
        return matcher;
      } else {
        llvm::Expected<ast::MacroMatch> match = parseMacroMatch();
        if (auto e = match.takeError()) {
          llvm::errs() << "failed to parse macro match in "
                          "macro matcher : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(*match);
      }
    }
  } else if (check(TokenKind::BraceOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "failed to parse marco matcher: eof");
      } else if (check(TokenKind::BraceClose)) {
        assert(eat(TokenKind::BraceClose));
        matcher.setKind(MacroMatcherKind::Brace);
        return matcher;
      } else {
        llvm::Expected<ast::MacroMatch> match = parseMacroMatch();
        if (auto e = match.takeError()) {
          llvm::errs() << "failed to parse macro match in "
                          "macro matcher : "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(*match);
      }
    }
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse marco matcher");
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse marco matcher");
}

llvm::Expected<ast::MacroRule> Parser::parseMacroRule() {
  Location loc = getLocation();
  MacroRule rule = {loc};

  llvm::Expected<ast::MacroMatcher> matcher = parseMacroMatcher();
  if (auto e = matcher.takeError()) {
    llvm::errs() << "failed to parse macro matcher in "
                    "macro rule : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  rule.setMatcher(*matcher);

  if (!check(TokenKind::FatArrow)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse => token in macro rule");
  }
  assert(eat(TokenKind::FatArrow));

  llvm::Expected<ast::MacroTranscriber> transcriber = parseMacroTranscriber();
  if (auto e = transcriber.takeError()) {
    llvm::errs() << "failed to parse macro trnscriber "
                    "in macro matcher : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  rule.setTranscriber(*transcriber);

  return rule;
}

llvm::Expected<ast::MacroRules> Parser::parseMacroRules() {
  Location loc = getLocation();
  MacroRules rules = {loc};

  llvm::Expected<ast::MacroRule> rule = parseMacroRule();
  if (auto e = rule.takeError()) {
    llvm::errs() << "failed to parse macro rule in "
                    "macro rules : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  rules.addRule(*rule);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse macro rules: eof");
    } else if (check(TokenKind::ParenClose)) {
      return rules;
    } else if (check(TokenKind::SquareClose)) {
      return rules;
    } else if (check(TokenKind::BraceClose)) {
      return rules;
    } else if (check(TokenKind::Semi) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Semi));
      return rules;
    } else if (check(TokenKind::Semi) && check(TokenKind::SquareClose, 1)) {
      assert(eat(TokenKind::Semi));
      return rules;
    } else if (check(TokenKind::Semi) && check(TokenKind::BraceClose, 1)) {
      assert(eat(TokenKind::Semi));
      return rules;
    } else if (check(TokenKind::Semi)) {
      assert(eat(TokenKind::Semi));
      llvm::Expected<ast::MacroRule> rule = parseMacroRule();
      if (auto e = rule.takeError()) {
        llvm::errs() << "failed to parse macro rule "
                        "in macro rules : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      rules.addRule(*rule);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse macro rules");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse macro rules");
}

llvm::Expected<ast::MacroRulesDef> Parser::parseMacroRulesDef() {
  Location loc = getLocation();
  MacroRulesDef def = {loc};

  if (check(TokenKind::ParenOpen)) {
    llvm::Expected<ast::MacroRules> rules = parseMacroRules();
    if (auto e = rules.takeError()) {
      llvm::errs() << "failed to parse macro rules in "
                      "macro rules def : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    def.setRules(*rules);
    if (!check(TokenKind::ParenClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse ) token in macro rules "
                               "def");
    assert(eat(TokenKind::ParenClose));
    if (!check(TokenKind::Semi))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse ; token in macro rules "
                               "def");
    assert(eat(TokenKind::Semi));
    def.setKind(MacroRulesDefKind::Paren);
    return def;
  } else if (check(TokenKind::SquareOpen)) {
    llvm::Expected<ast::MacroRules> rules = parseMacroRules();
    if (auto e = rules.takeError()) {
      llvm::errs() << "failed to parse macro rules in "
                      "macro rules def : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    def.setRules(*rules);
    if (!check(TokenKind::ParenClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse ] token in macro rules "
                               "def");
    assert(eat(TokenKind::ParenClose));
    if (!check(TokenKind::Semi))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse ; token in macro rules "
                               "def");
    assert(eat(TokenKind::Semi));
    def.setKind(MacroRulesDefKind::Square);
    return def;
  } else if (check(TokenKind::BraceOpen)) {
    llvm::Expected<ast::MacroRules> rules = parseMacroRules();
    if (auto e = rules.takeError()) {
      llvm::errs() << "failed to parse macro rules in "
                      "macro rules def : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    def.setRules(*rules);
    if (!check(TokenKind::ParenClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse } token in macro rules "
                               "def");
    assert(eat(TokenKind::ParenClose));
    def.setKind(MacroRulesDefKind::Brace);
    return def;
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse macro rules def");
}

llvm::Expected<std::shared_ptr<ast::MacroItem>>
Parser::parseMacroRulesDefinition() {
  Location loc = getLocation();
  MacroRulesDefinition def = {loc};

  if (!checkKeyWord(KeyWordKind::KW_MACRO_RULES))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse macro_rules keyword in macro "
                             "rules definition");
  assert(eatKeyWord(KeyWordKind::KW_MACRO_RULES));

  if (!check(TokenKind::Not))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ! token in macro rules "
                             "definition");
  assert(eat(TokenKind::Not));

  if (!check(TokenKind::Identifier))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse identifier token in macro "
                             "rules definition");

  def.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  llvm::Expected<ast::MacroRulesDef> rulesDef = parseMacroRulesDef();
  if (auto e = rulesDef.takeError()) {
    llvm::errs() << "failed to parse macro rules def "
                    "in macro rules definition : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  def.setDefinition(*rulesDef);

  return std::make_shared<MacroRulesDefinition>(def);
}

} // namespace rust_compiler::parser
