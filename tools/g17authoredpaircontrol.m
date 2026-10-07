// Native Metal control for the retained authored archive. A passive Frida hook
// must arm g17_capture_ready before the sole command can commit. This process
// does use Metal and is never evidence of below-Metal execution.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

__attribute__((visibility("default"))) volatile uint32_t g17_capture_ready = 0;

int main(int argc, const char **argv) { @autoreleasepool {
  if (argc != 4 || (argv[1][0]!='1' && argv[1][0]!='2') || argv[1][1]) return 2;
  unsigned bias=argv[1][0]-'0';
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  if (!dev) return 3;
  NSError *err=nil;
  NSData *libBytes=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
  if (!libBytes) return 4;
  dispatch_data_t dd=dispatch_data_create(libBytes.bytes,libBytes.length,nil,DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  id<MTLLibrary> lib=[dev newLibraryWithData:dd error:&err];
  if (!lib) { fprintf(stderr,"library: %s\n",err.description.UTF8String); return 5; }
  id<MTLFunction> fn=[lib newFunctionWithName:@"pt"];
  if (!fn) return 6;
  MTLBinaryArchiveDescriptor *ad=[MTLBinaryArchiveDescriptor new];
  ad.url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[3]]];
  id<MTLBinaryArchive> arc=[dev newBinaryArchiveWithDescriptor:ad error:&err];
  if (!arc) { fprintf(stderr,"archive: %s\n",err.description.UTF8String); return 7; }
  MTLComputePipelineDescriptor *pd=[MTLComputePipelineDescriptor new];
  pd.computeFunction=fn; pd.binaryArchives=@[arc];
  id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&err];
  if (!ps) { fprintf(stderr,"pipeline: %s\n",err.description.UTF8String); return 8; }
  uint32_t a[256],b[256],c[256];
  for(unsigned i=0;i<256;i++){a[i]=11*i+5;b[i]=7*i+3;c[i]=0xDEADBEEF;}
  id<MTLBuffer> A=[dev newBufferWithBytes:a length:sizeof a options:MTLResourceStorageModeShared];
  id<MTLBuffer> B=[dev newBufferWithBytes:b length:sizeof b options:MTLResourceStorageModeShared];
  id<MTLBuffer> C=[dev newBufferWithBytes:c length:sizeof c options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q=[dev newCommandQueue];
  id<MTLCommandBuffer> cb=g17_gpu_cb(q);
  id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  if (!A||!B||!C||!q||!cb||!enc) return 9;
  [enc setComputePipelineState:ps];
  [enc setBuffer:A offset:0 atIndex:0];
  [enc setBuffer:B offset:0 atIndex:1];
  [enc setBuffer:C offset:0 atIndex:2];
  [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [enc endEncoding];
  for(unsigned i=0;i<5000 && g17_capture_ready!=0x17C0DE02;i++)usleep(1000);
  if(g17_capture_ready!=0x17C0DE02){fprintf(stderr,"REFUSED: passive hook not ready\n");return 10;}
  [cb commit];[cb waitUntilCompleted];
  uint32_t *o=(uint32_t*)C.contents;
  unsigned exact=0,guards=0;
  for(unsigned i=0;i<64;i++)exact+=(o[i]==3*i+bias);
  for(unsigned i=64;i<256;i++)guards+=(o[i]==0xDEADBEEF);
  printf("{\"bias\":%u,\"status\":%lu,\"exact_64\":%u,\"sentinel_192\":%u}\n",
         bias,(unsigned long)cb.status,exact,guards);
  return cb.status==MTLCommandBufferStatusCompleted && exact==64 && guards==192 ? 0:11;
} }
