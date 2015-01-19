#include "CudaFramework/General/Utilities.hpp"


std::string Utilities::cutStringTillScope(std::string s){
      s.erase( 0, (size_t)s.find_last_of("::")+1);
      return s;
}
