// Metal reference for submission tracing, NOT an authored-code execution proof.
// One dispatch per process; all 64 outputs and both guard regions are checked.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(int argc, const char **argv) { @autoreleasepool {
    if(argc!=2 || (strcmp(argv[1],"1") && strcmp(argv[1],"2")))return 2;
    unsigned bias=argv[1][0]-'0';
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    if(!dev)return 3;
    NSString *src=[NSString stringWithFormat:@"kernel void k(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i]=3u*i+%uu; }",bias];
    NSError *err=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
    if(!lib){fprintf(stderr,"library: %s\n",err.description.UTF8String);return 4;}
    id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];
    if(!ps){fprintf(stderr,"pipeline: %s\n",err.description.UTF8String);return 5;}
    id<MTLBuffer> buffer=[dev newBufferWithLength:96*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q=[dev newCommandQueue];
    if(!buffer || !q)return 6;
    uint32_t *p=buffer.contents;
    for(unsigned i=0;i<96;i++)p[i]=0xDEADBEEF;
    id<MTLCommandBuffer> cb=g17_gpu_cb(q);
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    if(!cb || !enc)return 7;
    [enc setComputePipelineState:ps];
    [enc setBuffer:buffer offset:16*sizeof(uint32_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted || cb.error){
        fprintf(stderr,"command status=%lu error=%s\n",(unsigned long)cb.status,cb.error.description.UTF8String);return 8;
    }
    unsigned correct=0,guards=0;
    for(unsigned i=0;i<64;i++)correct+=(p[16+i]==3*i+bias);
    for(unsigned i=0;i<16;i++)guards+=(p[i]==0xDEADBEEF)+(p[80+i]==0xDEADBEEF);
    printf("{\"producer\":\"Apple Metal reference\",\"bias\":%u,\"outputs_correct\":%u,\"outputs_total\":64,\"guards_correct\":%u,\"guards_total\":32,\"status\":%lu,\"output\":[",bias,correct,guards,(unsigned long)cb.status);
    for(unsigned i=0;i<64;i++)printf("%s%u",i?",":"",p[16+i]);
    puts("]}");
    return correct==64 && guards==32 ? 0:9;
} }
