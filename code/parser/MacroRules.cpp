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
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

StringResult<ast::MacroMatch> Parser::parseMacroMatch() {
  Location loc = getLocation();
  MacroMatch match = {loc};

  if (check(TokenKind::Dollar) && check(TokenKind::ParenOpen, 1)) {
    eat(TokenKind::Dollar);
    eat(TokenKind::ParenOpen);
    while (true) {
      if (getToken().getKind() == TokenKind::ParenClose) {
      } else if (getToken().getKind() == TokenKind::Eof) {
      }
      StringResult<ast::MacroMatch> match = parseMacroMatch();
    }
    // MacroMatch+
  } else if (check(TokenKind::Dollar) && check(TokenKind::Underscore, 1)) {
    //
  } else if (check(TokenKind::Dollar) && check(TokenKind::Identifier, 1) &&
             getToken(1).getIdentifier().isRawIdentifier()) {
    //
  } else if (check(TokenKind::Dollar) && check(TokenKind::Identifier, 1)) {
    //
  } else if (check(TokenKind::Dollar) && check(TokenKind::Keyword, 1)) {
    //
  } else if (checkDelimiters()) {
    // MacroMatcher
    StringResult<ast::MacroMatcher> matcher = parseMacroMatcher();
  } else if (!checkDelimiters()) {
    match.setToken(getToken());
    return StringResult<ast::MacroMatch>(match);
  }

  // error
}

StringResult<ast::MacroRepSep> Parser::parseMacroRepSep() {
  Location loc = getLocation();

  MacroRepSep sep = {loc};

  if (checkDelimiters()) {
    return StringResult<ast::MacroRepSep>("failed to parse macro rep sep");
  } else if (check(TokenKind::Star)) {
    return StringResult<ast::MacroRepSep>("failed to parse macro rep sep");
  } else if (check(TokenKind::Plus)) {
    return StringResult<ast::MacroRepSep>("failed to parse macro rep sep");
  } else if (check(TokenKind::QMark)) {
    return StringResult<ast::MacroRepSep>("failed to parse macro rep sep");
  }

  sep.setToken(getToken());
  return StringResult<ast::MacroRepSep>(sep);
}

StringResult<ast::MacroRepOp> Parser::parseMacroRepOp() {
  Location loc = getLocation();
  MacroRepOp op = {loc};

  if (check(TokenKind::Star)) {
    op.setKind(MacroRepOpKind::Star);
  } else if (check(TokenKind::Plus)) {
    op.setKind(MacroRepOpKind::Plus);
  } else if (check(TokenKind::QMark)) {
    op.setKind(MacroRepOpKind::Qmark);
  } else {
    return StringResult<ast::MacroRepOp>("failed to parse marco rep op");
  }

  return StringResult<ast::MacroRepOp>(op);
}

StringResult<ast::MacroFragSpec> Parser::parseMacroFragSpec() {
  Location loc = getLocation();
  MacroFragSpec frag = {loc};

  if (!checkIdentifier())
    return StringResult<ast::MacroFragSpec>("failed to parse marco frag spec");

  MacroFragSpecKind k =
      StringSwitch<MacroFragSpecKind>(getToken().getIdentifier().toString())
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

  return StringResult<ast::MacroFragSpec>(frag);
}

StringResult<ast::MacroTranscriber> Parser::parseMacroTranscriber() {
  Location loc = getLocation();
  MacroTranscriber transcriber = {loc};

  StringResult<std::shared_ptr<ast::DelimTokenTree>> tree =
      parseDelimTokenTree();
  if (!tree) {
    llvm::errs() << "failed to parse delim token tree in macro transcribe: "
                 << tree.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  transcriber.setTree(tree.getValue());

  return StringResult<ast::MacroTranscriber>(transcriber);
}

StringResult<ast::MacroMatcher> Parser::parseMacroMatcher() {
  Location loc = getLocation();
  MacroMatcher matcher = {loc};

  if (check(TokenKind::ParenOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<ast::MacroMatcher>(
            "failed to parse marco matcher: eof");
      } else if (check(TokenKind::ParenClose)) {
        assert(eat(TokenKind::ParenClose));
        matcher.setKind(MacroMatcherKind::Paren);
        return StringResult<ast::MacroMatcher>(matcher);
      } else {
        StringResult<ast::MacroMatch> match = parseMacroMatch();
        if (!match) {
          llvm::errs() << "failed to parse macro match in macro matcher: "
                       << match.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(match.getValue());
      }
    }
  } else if (check(TokenKind::SquareOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<ast::MacroMatcher>(
            "failed to parse marco matcher: eof");
      } else if (check(TokenKind::SquareClose)) {
        assert(eat(TokenKind::SquareClose));
        matcher.setKind(MacroMatcherKind::Square);
        return StringResult<ast::MacroMatcher>(matcher);
      } else {
        StringResult<ast::MacroMatch> match = parseMacroMatch();
        if (!match) {
          llvm::errs() << "failed to parse macro match in macro matcher: "
                       << match.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(match.getValue());
      }
    }
  } else if (check(TokenKind::BraceOpen)) {
    while (true) {
      if (check(TokenKind::Eof)) {
        return StringResult<ast::MacroMatcher>(
            "failed to parse marco matcher: eof");
      } else if (check(TokenKind::BraceClose)) {
        assert(eat(TokenKind::BraceClose));
        matcher.setKind(MacroMatcherKind::Brace);
        return StringResult<ast::MacroMatcher>(matcher);
      } else {
        StringResult<ast::MacroMatch> match = parseMacroMatch();
        if (!match) {
          llvm::errs() << "failed to parse macro match in macro matcher: "
                       << match.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        matcher.addMatch(match.getValue());
      }
    }
  } else {
    return StringResult<ast::MacroMatcher>("failed to parse marco matcher");
  }
  return StringResult<ast::MacroMatcher>("failed to parse marco matcher");
}

StringResult<ast::MacroRule> Parser::parseMacroRule() {
  Location loc = getLocation();
  MacroRule rule = {loc};

  StringResult<ast::MacroMatcher> matcher = parseMacroMatcher();
  if (!matcher) {
    llvm::errs() << "failed to parse macro matcher in macro rule: "
                 << matcher.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  rule.setMatcher(matcher.getValue());

  if (!check(TokenKind::FatArrow)) {
    return StringResult<ast::MacroRule>(
        "failed to parse => token in macro rule");
  }
  assert(eat(TokenKind::FatArrow));

  StringResult<ast::MacroTranscriber> transcriber = parseMacroTranscriber();
  if (!transcriber) {
    llvm::errs() << "failed to parse macro transcriber in macro rule: "
                 << transcriber.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  rule.setTranscriber(transcriber.getValue());

  return StringResult<ast::MacroRule>(rule);
}

StringResult<ast::MacroRules> Parser::parseMacroRules() {
  Location loc = getLocation();
  MacroRules rules = {loc};

  StringResult<ast::MacroRule> rule = parseMacroRule();
  if (!rule) {
    llvm::errs() << "failed to parse macro rule in macro rules: "
                 << rule.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  rules.addRule(rule.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::MacroRules>("failed to parse macro rules: eof");
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::SquareClose)) {
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::BraceClose)) {
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::Semi) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Semi));
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::Semi) && check(TokenKind::SquareClose, 1)) {
      assert(eat(TokenKind::Semi));
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::Semi) && check(TokenKind::BraceClose, 1)) {
      assert(eat(TokenKind::Semi));
      return StringResult<ast::MacroRules>(rules);
    } else if (check(TokenKind::Semi)) {
      assert(eat(TokenKind::Semi));
      StringResult<ast::MacroRule> rule = parseMacroRule();
      if (!rule) {
        llvm::errs() << "failed to parse macro rule in macro rules: "
                     << rule.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      rules.addRule(rule.getValue());
      // FIXME: return rules?
    } else {
      return StringResult<ast::MacroRules>("failed to parse macro rules");
    }
  }
  return StringResult<ast::MacroRules>("failed to parse macro rules");
}

StringResult<ast::MacroRulesDef> Parser::parseMacroRulesDef() {
  Location loc = getLocation();
  MacroRulesDef def = {loc};

  if (check(TokenKind::ParenOpen)) {
    StringResult<ast::MacroRules> rules = parseMacroRules();
    if (!rules) {
      llvm::errs() << "failed to parse macro rules in macro rules def: "
                   << rules.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    def.setRules(rules.getValue());
    if (!check(TokenKind::ParenClose))
      return StringResult<ast::MacroRulesDef>(
          "failed to parse ) token in macro rules "
          "def");
    assert(eat(TokenKind::ParenClose));
    if (!check(TokenKind::Semi))
      return StringResult<ast::MacroRulesDef>(
          "failed to parse ; token in macro rules "
          "def");
    assert(eat(TokenKind::Semi));
    def.setKind(MacroRulesDefKind::Paren);
    return StringResult<ast::MacroRulesDef>(def);
  } else if (check(TokenKind::SquareOpen)) {
    StringResult<ast::MacroRules> rules = parseMacroRules();
    if (!rules) {
      llvm::errs() << "failed to parse macro rules in macro rules def: "
                   << rules.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    def.setRules(rules.getValue());
    if (!check(TokenKind::ParenClose))
      return StringResult<ast::MacroRulesDef>(
          "failed to parse ] token in macro rules "
          "def");
    assert(eat(TokenKind::ParenClose));
    if (!check(TokenKind::Semi))
      return StringResult<ast::MacroRulesDef>(
          "failed to parse ; token in macro rules "
          "def");
    assert(eat(TokenKind::Semi));
    def.setKind(MacroRulesDefKind::Square);
    return StringResult<ast::MacroRulesDef>(def);
  } else if (check(TokenKind::BraceOpen)) {
    StringResult<ast::MacroRules> rules = parseMacroRules();
    if (!rules) {
      llvm::errs() << "failed to parse macro rules in macro rules def: "
                   << rules.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    def.setRules(rules.getValue());
    if (!check(TokenKind::ParenClose))
      return StringResult<ast::MacroRulesDef>(
          "failed to parse } token in macro rules "
          "def");
    assert(eat(TokenKind::ParenClose));
    def.setKind(MacroRulesDefKind::Brace);
    return StringResult<ast::MacroRulesDef>(def);
  }
  return StringResult<ast::MacroRulesDef>("failed to parse macro rules def");
}

StringResult<std::shared_ptr<ast::Item>> Parser::parseMacroRulesDefinition() {
  Location loc = getLocation();
  MacroRulesDefinition def = {loc};

  if (!checkKeyWord(KeyWordKind::KW_MACRO_RULES))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse macro_rules keyword in macro "
        "rules definition");
  assert(eatKeyWord(KeyWordKind::KW_MACRO_RULES));

  if (!check(TokenKind::Not))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse ! token in macro rules "
        "definition");
  assert(eat(TokenKind::Not));

  if (!check(TokenKind::Identifier))
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse identifier token in macro "
        "rules definition");

  def.setIdentifier(getToken().getIdentifier());
  assert(eat(TokenKind::Identifier));

  StringResult<ast::MacroRulesDef> rulesDef = parseMacroRulesDef();
  if (!rulesDef) {
    llvm::errs() << "failed to parse macro rules ef in macro rules definition: "
                 << rulesDef.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  def.setDefinition(rulesDef.getValue());

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<MacroRulesDefinition>(def));
}

} // namespace rust_compiler::parser
