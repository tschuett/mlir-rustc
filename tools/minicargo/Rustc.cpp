#include "Rustc.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::minicargo {

void invokeRustC(toml::Toml toml) {
  llvm::SmallVector<char, 128> cwd;

  std::error_code ec = llvm::sys::fs::current_path(cwd);

  if (ec)
    return;

  llvm::outs() << cwd << "\n";

  llvm::sys::path::append(cwd, "tools");
  llvm::sys::path::append(cwd, "rustc");
  llvm::sys::path::append(cwd, "rustc");

  llvm::outs() << cwd << "\n";
}

} // namespace rust_compiler::minicargo
