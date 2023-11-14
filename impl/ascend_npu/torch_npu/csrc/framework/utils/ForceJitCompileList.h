#pragma once

#include <string>
#include <set>
#include <vector>

using std::string;
using std::vector;

namespace at_npu {
namespace native {

class ForceJitCompileList {
public:
  static ForceJitCompileList& GetInstance();
  void RegisterJitlist(const std::string& blacklist);
  bool Inlist(const std::string& opName) const;
  void DisplayJitlist() const;
  ~ForceJitCompileList() = default;
private:
  ForceJitCompileList() {}
  std::set<std::string> jit_list_;
};

}
}