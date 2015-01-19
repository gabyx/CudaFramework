#include "CudaFramework/CudaModern/CudaDeviceGroup.hpp"
#include "CudaFramework/CudaModern/CudaContextGroup.hpp"


namespace utilCuda{


    // Leave this order!! contextGroup gets deleted first, then deviceGroup
    // Global Device Group
    std::unique_ptr<DeviceGroup> deviceGroup;

    // Global Context Group
    //    std::unique_ptr<ContextGroup> contextGroup;

};

