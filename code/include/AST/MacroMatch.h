#pragma once

#include "AST/AST.h"
#include "AST/MacroFragSpec.h"
#include "AST/MacroRepOp.h"
#include "AST/MacroRepSep.h"
#include "Lexer/Identifier.h"
#include "Lexer/Token.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class MacroMatcher;

enum class MacroMatchKind { Token, MacroMatcher, Identifier, Recursive };

class MacroMatch : public Node {
  MacroMatchKind kind;
  std::optional<lexer::Token> token;
  std::optional<MacroFragSpec> spec;
  Identifier identifier;
  Identifier rawIdentifier;
  std::shared_ptr<MacroMatcher> matcher;
  std::optional<MacroRepSep> repSep;
  std::optional<MacroRepOp> repOp;
  bool underscore = false;
  std::vector<MacroMatch> macroMatchers;

public:
  MacroMatch(Location loc) : Node(loc) {}

  Identifier getIdentifier() const { return identifier; }

  void setToken(const lexer::Token &tok) {
    token = tok;
    kind = MacroMatchKind::Token;
  }

  void setKeyWord(const Token &tok) {
    token = tok;
    kind = MacroMatchKind::Identifier;
  }

  void setIdentifier(const Identifier &id) {
    identifier = id;
    kind = MacroMatchKind::Identifier;
  }

  void setRawIdentifier(const Identifier &id) {
    rawIdentifier = id;
    kind = MacroMatchKind::Identifier;
  }

  void setUnderScore() {
    underscore = true;
  }

  void setFragSpec(const MacroFragSpec &_spec) {
    spec = _spec;
    kind = MacroMatchKind::Identifier;
  }

  void setMacroMatcher(std::shared_ptr<MacroMatcher> m) {
    matcher = m;
    kind = MacroMatchKind::MacroMatcher;
  }

  void setMacroRepSep(const MacroRepSep &sep) {
    repSep = sep;
    kind = MacroMatchKind::Recursive;
  }

  void setMacroRepOp(const MacroRepOp &op) {
    repOp = op;
    kind = MacroMatchKind::Recursive;
  }

  void addMacroMatch(const MacroMatch& m) {
    macroMatchers.push_back(m);
  }
};

} // namespace rust_compiler::ast
