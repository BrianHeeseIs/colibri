#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main(){ @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if(!d){ printf("  NO METAL DEVICE\n"); return 1; }
  printf("  name                     : %s\n", [[d name] UTF8String]);
  printf("  unified memory           : %s\n", d.hasUnifiedMemory?"YES":"no");
  printf("  max threadgroup mem      : %lu bytes\n", (unsigned long)d.maxThreadgroupMemoryLength);
  printf("  max threads/threadgroup  : %lu\n", (unsigned long)d.maxThreadsPerThreadgroup.width);
  printf("  recommended max working  : %.1f GB\n", d.recommendedMaxWorkingSetSize/1e9);
  printf("  registryID               : %llu\n", d.registryID);
  if (@available(macOS 13.0,*)) printf("  supports family Metal3   : %s\n", [d supportsFamily:MTLGPUFamilyMetal3]?"YES":"no");
  for (int a=9; a>=4; a--) if ([d supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1+a-1)]) { printf("  highest GPUFamily Apple  : %d\n", a); break; }
  return 0; }}
