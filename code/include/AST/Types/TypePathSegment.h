#pragma once

#include "AST/AST.h"
#include "AST/GenericArgs.h"
#include "AST/PathIdentSegment.h"
#include "AST/Types/TypePathFn.h"

#include <optional>
#include <variant>
#include <vector>

namespace rust_compiler::ast::types {

class TypePathSegment final : public Node {
  PathIdentSegment pathIdentSegment;
  bool doubleColon;
  std::optional<std::variant<GenericArgs, TypePathFn>> tail;

public:
  TypePathSegment(Location loc) : Node(loc), pathIdentSegment(loc) {}

  void setSegment(const PathIdentSegment &seg) { pathIdentSegment = seg; }

  void setDoubleColon() { doubleColon = true; }

  void setGenericArgs(const GenericArgs &a) { tail = a; }

  void setTypePathFn(const TypePathFn &f) { tail = f; }

  PathIdentSegment getSegment() const { return pathIdentSegment; }

  bool hasGenerics() const {
    return tail.has_value() && std::holds_alternative<GenericArgs>(*tail);
  }

  bool hasTypeFunction() const {
    return tail.has_value() && std::holds_alternative<TypePathFn>(*tail);
  }

  TypePathFn getTypePathFn() const { return std::get<TypePathFn>(*tail); }

  GenericArgs getGenericArgs() const { return std::get<GenericArgs>(*tail); }
};

} // namespace rust_compiler::ast::types
