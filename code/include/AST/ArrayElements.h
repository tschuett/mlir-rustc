#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

enum class ArrayElementsKind { List, Repeated };

class ArrayElements : public Node {
  ArrayElementsKind kind;
  std::shared_ptr<Expression> count;
  std::shared_ptr<Expression> value;
  std::vector<std::shared_ptr<Expression>> elements;

public:
  ArrayElements(Location loc) : Node(loc) {}

  void setKind(ArrayElementsKind k) { kind = k; }
  void setValue(std::shared_ptr<Expression> v) { value = v; }
  void setCount(std::shared_ptr<Expression> e) { count = e; }
  void addElement(std::shared_ptr<Expression> e) { elements.push_back(e); }
};

} // namespace rust_compiler::ast
