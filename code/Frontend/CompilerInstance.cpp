#include "Frontend/CompilerInstance.h"

#include "llvm/Support/raw_ostream.h"

#include <llvm/Support/Path.h>
#include <memory>
#include <system_error>

namespace rust_compiler::frontend {

CompilerInstance::CompilerInstance() : invocation(new CompilerInvocation()) {}

std::unique_ptr<llvm::raw_pwrite_stream>
CompilerInstance::createDefaultOutputFile(llvm::StringRef baseInput,
                                          llvm::StringRef extension) {
  if (baseInput.ends_with(".rs")) {
    llvm::SmallVector<char, 128> objectFile{baseInput.begin(), baseInput.end()};
    llvm::sys::path::replace_extension(objectFile, extension);

    std::string object = {objectFile.begin(), objectFile.end()};

    std::error_code EC;
    
    llvm::raw_fd_stream s = {object, EC};

    // Creates the file descriptor for the output file
    std::unique_ptr<llvm::raw_fd_ostream> os;

    std::error_code error;
    os.reset(new llvm::raw_fd_ostream(object, EC));
    if (EC) {
      // handle error
      llvm::errs() << "faile to open file: " << EC.message() << "\n";
    }

    return os;
  }

  return nullptr;
}

} // namespace rust_compiler::frontend
