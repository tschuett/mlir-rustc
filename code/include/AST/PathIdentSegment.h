#pragma once

#include "AST/AST.h"
#include "Lexer/Identifier.h"

#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

enum class PathIdentSegmentKind {
  Identifier,
  super,
  self,
  Self,
  crate,
  dollarCrate
};

class PathIdentSegment : public Node {
  PathIdentSegmentKind kind;
  Identifier identifier;

public:
  PathIdentSegment(Location loc) : Node(loc) {}

  PathIdentSegmentKind getKind() const { return kind; }

  void setIdentifier(const Identifier &i) {
    kind = PathIdentSegmentKind::Identifier;
    identifier = i;
  }

  void setSuper() { kind = PathIdentSegmentKind::super; }
  void setSelfValue() { kind = PathIdentSegmentKind::self; }
  void setSelfType() { kind = PathIdentSegmentKind::Self; }
  void setCrate() { kind = PathIdentSegmentKind::crate; }
  void setDollarCrate() { kind = PathIdentSegmentKind::dollarCrate; }

  Identifier getIdentifier() const { return identifier; }

  std::string toString() const {
    switch (kind) {
    case PathIdentSegmentKind::Identifier:
      return identifier.toString();
    case PathIdentSegmentKind::super:
      return "super";
    case PathIdentSegmentKind::self:
      return "self";
    case PathIdentSegmentKind::Self:
      return "Self";
    case PathIdentSegmentKind::crate:
      return "crate";
    case PathIdentSegmentKind::dollarCrate:
      return "$crate";
    }
  }
};

} // namespace rust_compiler::ast
