#pragma once

#include "AST/AST.h"

#include <string_view>

namespace rust_compiler::ast {

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
  std::string identifier;

public:
  PathIdentSegment(Location loc) : Node(loc) {}

  PathIdentSegmentKind getKind() const { return kind; }

  void setIdentifier(std::string_view i) {
    kind = PathIdentSegmentKind::Identifier;
    identifier = i;
  }

  void setSuper() { kind = PathIdentSegmentKind::super; }
  void setSelfValue() { kind = PathIdentSegmentKind::self; }
  void setSelfType() { kind = PathIdentSegmentKind::Self; }
  void setCrate() { kind = PathIdentSegmentKind::crate; }
  void setDollarCrate() { kind = PathIdentSegmentKind::dollarCrate; }

  std::string getIdentifier() const { return identifier; }

  std::string toString() const {
    switch(kind) {
    case PathIdentSegmentKind::Identifier:
      return identifier;
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
