#pragma once

namespace rust_compiler::analysis {

class MemorySSA;

class MemorySSAWalker {
public:
  MemorySSAWalker(MemorySSA *);

private:
  MemorySSA *MSSA;
};

} // namespace rust_compiler::analysis
