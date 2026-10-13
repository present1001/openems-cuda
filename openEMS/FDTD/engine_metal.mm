#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "engine_metal.h"
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <climits>

static const char* kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Grid {uint nx, ny, nz, cells;};
inline uint ix(uint c,uint x,uint y,uint z,constant Grid& g) {
 return c*g.cells+(x*g.ny+y)*g.nz+z;
}
kernel void volt_update(device float* v [[buffer(0)]],
 device const float* h [[buffer(1)]],
 device const float* vv [[buffer(2)]],
 device const float* vi [[buffer(3)]],
 constant Grid& g [[buffer(4)]],
 uint n [[thread_position_in_grid]]) {
 if(n>=g.cells) return;
 uint z=n%g.nz,y=(n/g.nz)%g.ny,x=n/(g.nz*g.ny);
 uint xm=x?x-1:0,ym=y?y-1:0,zm=z?z-1:0;
 uint a=ix(0,x,y,z,g),b=ix(1,x,y,z,g),c=ix(2,x,y,z,g);
 float a1=v[a]*vv[a]+vi[a]*(h[ix(2,x,y,z,g)]-h[ix(2,x,ym,z,g)]-h[ix(1,x,y,z,g)]+h[ix(1,x,y,zm,g)]);
 float b1=v[b]*vv[b]+vi[b]*(h[ix(0,x,y,z,g)]-h[ix(0,x,y,zm,g)]-h[ix(2,x,y,z,g)]+h[ix(2,xm,y,z,g)]);
 float c1=v[c]*vv[c]+vi[c]*(h[ix(1,x,y,z,g)]-h[ix(1,xm,y,z,g)]-h[ix(0,x,y,z,g)]+h[ix(0,x,ym,z,g)]);
 v[a]=a1;v[b]=b1;v[c]=c1;
}
kernel void curr_update(device float* h [[buffer(0)]],
 device const float* v [[buffer(1)]],
 device const float* ii [[buffer(2)]],
 device const float* iv [[buffer(3)]],
 constant Grid& g [[buffer(4)]],
 uint n [[thread_position_in_grid]]) {
 uint count=(g.nx-1)*(g.ny-1)*(g.nz-1);
 if(n>=count)return;
 uint z=n%(g.nz-1),y=(n/(g.nz-1))%(g.ny-1),x=n/((g.nz-1)*(g.ny-1));
 uint a=ix(0,x,y,z,g),b=ix(1,x,y,z,g),c=ix(2,x,y,z,g);
 float a1=h[a]*ii[a]+iv[a]*(v[ix(2,x,y,z,g)]-v[ix(2,x,y+1,z,g)]-v[ix(1,x,y,z,g)]+v[ix(1,x,y,z+1,g)]);
 float b1=h[b]*ii[b]+iv[b]*(v[ix(0,x,y,z,g)]-v[ix(0,x,y,z+1,g)]-v[ix(2,x,y,z,g)]+v[ix(2,x+1,y,z,g)]);
 float c1=h[c]*ii[c]+iv[c]*(v[ix(1,x,y,z,g)]-v[ix(1,x+1,y,z,g)]-v[ix(0,x,y,z,g)]+v[ix(0,x,y+1,z,g)]);
 h[a]=a1;h[b]=b1;h[c]=c1;
}
)METAL";

struct MetalGrid {uint32_t nx,ny,nz,cells;};
struct Engine_Metal::Impl {
 id<MTLDevice> device;
 id<MTLCommandQueue> queue;
 id<MTLComputePipelineState> eKernel,hKernel;
 id<MTLBuffer> v,h,vv,vi,ii,iv;
 MetalGrid grid;
 size_t bytes;
};
Engine_Metal* Engine_Metal::New(const Operator* op) {
 Engine_Metal* e=new Engine_Metal(op);
 try {e->Init();}catch(...) {delete e;throw;}
 return e;
}
Engine_Metal::Engine_Metal(const Operator* op):Engine(op),impl(nullptr) {m_type=METAL;}
Engine_Metal::~Engine_Metal(){Reset();}
void Engine_Metal::Init(){
 Engine::Init();
 try {
  impl=new Impl();
  impl->device=MTLCreateSystemDefaultDevice();
  if(!impl->device)throw std::runtime_error("Metal device unavailable");
  impl->queue=[impl->device newCommandQueue];
  NSError* err=nil;
  id<MTLLibrary> lib=[impl->device newLibraryWithSource:[NSString stringWithUTF8String:kShader] options:nil error:&err];
  if(!lib)throw std::runtime_error([[err localizedDescription] UTF8String]);
  impl->eKernel=[impl->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"volt_update"] error:&err];
  if(!impl->eKernel)throw std::runtime_error([[err localizedDescription] UTF8String]);
  impl->hKernel=[impl->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"curr_update"] error:&err];
  if(!impl->hKernel)throw std::runtime_error([[err localizedDescription] UTF8String]);
  size_t cells=(size_t)numLines[0]*numLines[1]*numLines[2];
  if(cells>UINT32_MAX)throw std::runtime_error("Metal requires <= 2^32 cells");
  impl->grid={numLines[0],numLines[1],numLines[2],(uint32_t)cells};
  impl->bytes=3*cells*sizeof(FDTD_FLOAT);
  auto alloc=[&]()->id<MTLBuffer>{id<MTLBuffer> b=[impl->device newBufferWithLength:impl->bytes options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("Metal buffer allocation failed");return b;};
  impl->v=alloc();impl->h=alloc();impl->vv=alloc();impl->vi=alloc();impl->ii=alloc();impl->iv=alloc();
  memcpy([impl->vv contents],Op->vv_ptr->data(),impl->bytes);
  memcpy([impl->vi contents],Op->vi_ptr->data(),impl->bytes);
  memcpy([impl->ii contents],Op->ii_ptr->data(),impl->bytes);
  memcpy([impl->iv contents],Op->iv_ptr->data(),impl->bytes);
 }catch(...){Reset();throw;}
}
void Engine_Metal::Reset(){delete impl;impl=nullptr;Engine::Reset();}
void Engine_Metal::Step(bool voltage){
 if(!impl)throw std::runtime_error("Metal not initialized");
 const MetalGrid& g=impl->grid;
 size_t n=voltage?g.cells:(size_t)(g.nx-1)*(g.ny-1)*(g.nz-1);
 if(!n)return;
 memcpy([impl->v contents],volt_ptr->data(),impl->bytes);
 memcpy([impl->h contents],curr_ptr->data(),impl->bytes);
 id<MTLCommandBuffer> cb=[impl->queue commandBuffer];
 id<MTLComputeCommandEncoder> cmd=[cb computeCommandEncoder];
 id<MTLComputePipelineState> pipeline=voltage?impl->eKernel:impl->hKernel;
 [cmd setComputePipelineState:pipeline];
 [cmd setBuffer:voltage?impl->v:impl->h offset:0 atIndex:0];
 [cmd setBuffer:voltage?impl->h:impl->v offset:0 atIndex:1];
 [cmd setBuffer:voltage?impl->vv:impl->ii offset:0 atIndex:2];
 [cmd setBuffer:voltage?impl->vi:impl->iv offset:0 atIndex:3];
 [cmd setBytes:&g length:sizeof(g) atIndex:4];
 NSUInteger threads=MIN((NSUInteger)256,[pipeline maxTotalThreadsPerThreadgroup]);
 [cmd dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
 [cmd endEncoding];
 [cb commit];[cb waitUntilCompleted];
 if([cb status]!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal command execution failed");
 memcpy(voltage?volt_ptr->data():curr_ptr->data(),
        [voltage?impl->v:impl->h contents],impl->bytes);
}
bool Engine_Metal::IterateTS(unsigned int count){
 for(unsigned int i=0;i<count;++i){
  DoPreVoltageUpdates();Step(true);DoPostVoltageUpdates();Apply2Voltages();
  DoPreCurrentUpdates();Step(false);DoPostCurrentUpdates();Apply2Current();
  ++numTS;
 }
 return true;
}
