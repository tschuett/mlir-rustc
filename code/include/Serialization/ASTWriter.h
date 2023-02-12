#pragma once

#include "AST/Crate.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitstream/BitstreamWriter.h>
#include <memory>
#include <string_view>

namespace rust_compiler::serialization {

class ASTWriter {
  /// The bitstream writer used to emit this precompiled header.
  llvm::BitstreamWriter &stream;

  /// The buffer associated with the bitstream.
  const llvm::SmallVectorImpl<char> &buffer;

public:
  ASTWriter(llvm::BitstreamWriter &stream, llvm::SmallVectorImpl<char> &buffer)
      : stream(stream), buffer(buffer) {}

  void writeAst(std::shared_ptr<ast::Crate>, std::string_view outputFile);
};

} // namespace rust_compiler::serialization
