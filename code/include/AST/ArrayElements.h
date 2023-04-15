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
  ArrayElementsKind getKind() const { return kind; }

  std::shared_ptr<Expression> getCount() const { return count; }
  std::shared_ptr<Expression> getValue() const { return value; }

  std::vector<std::shared_ptr<Expression>> &getElements()  {
    return elements;
  }
};

} // namespace rust_compiler::ast
