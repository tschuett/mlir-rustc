#pragma once

namespace rust_compiler::ast {
class Trait;
}

namespace rust_compiler::tyctx {

/// a reference to a Trait
class TraitReference {

public:
  ast::Trait *getTrait() const { return trait; }

private:
  ast::Trait *trait;
};

} // namespace rust_compiler::tyctx
