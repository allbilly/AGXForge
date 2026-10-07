// G13 machine-code carriers through Metal. Apple compiles the carrier only;
// archive misses are fatal, so dispatch cannot silently compile its source.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>

@interface AFMetal : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSMutableArray *resources;
@property(nonatomic, strong) id<MTLCommandBuffer> active;
@property(nonatomic) BOOL stopped;
@end
@implementation AFMetal
@end

static _Thread_local char last_error[2048];
static int fail(const char *phase, NSError *error) {
    snprintf(last_error, sizeof last_error, "%s: %s", phase,
             error ? error.localizedDescription.UTF8String : "invalid state");
    return -1;
}
const char *am_error(void) { return last_error; }

void *am_open(void) {
    @autoreleasepool {
        AFMetal *ctx = [AFMetal new];
        ctx.device = MTLCreateSystemDefaultDevice();
        // An ISA family, not just Apple silicon: reject M1 Pro/Max/Ultra and G14+.
        if (!ctx.device || ![ctx.device.name isEqualToString:@"Apple M1"]) {
            fail("expected base Apple M1 / G13G", nil); return NULL;
        }
        ctx.queue = [ctx.device newCommandQueue];
        ctx.resources = [NSMutableArray new];
        if (!ctx.queue) { fail("command queue", nil); return NULL; }
        return (__bridge_retained void *)ctx;
    }
}
int am_close(void *opaque) {
    @autoreleasepool {
        // A submitted command buffer retains the resources it uses. Never wait
        // without a bound or release them manually after a completion timeout.
        (void)(__bridge_transfer AFMetal *)opaque;
        return 0;
    }
}
const char *am_name(void *opaque) {
    return ((__bridge AFMetal *)opaque).device.name.UTF8String;
}
void *am_alloc(void *opaque, uint64_t size) {
    @autoreleasepool {
        AFMetal *ctx = (__bridge AFMetal *)opaque;
        if (ctx.stopped || !size || size > ctx.device.maxBufferLength) {
            fail("buffer extent or stopped executor", nil); return NULL;
        }
        id<MTLBuffer> buffer = [ctx.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer || !buffer.contents) { fail("shared buffer", nil); return NULL; }
        [ctx.resources addObject:buffer];
        return (__bridge void *)buffer;
    }
}
void *am_map(void *buffer) { return ((__bridge id<MTLBuffer>)buffer).contents; }
uint64_t am_address(void *buffer) { return ((__bridge id<MTLBuffer>)buffer).gpuAddress; }

int am_carrier(void *opaque, const char *library_path, const char *archive_path) {
    @autoreleasepool {
        AFMetal *ctx = (__bridge AFMetal *)opaque;
        NSError *error = nil;
        id<MTLLibrary> library = [ctx.device newLibraryWithURL:
            [NSURL fileURLWithPath:@(library_path)] error:&error];
        if (!library) return fail("carrier library", error);
        MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
        pd.computeFunction = [library newFunctionWithName:@"carrier"];
        if (!pd.computeFunction) return fail("carrier function", nil);
        id<MTLBinaryArchive> archive = [ctx.device newBinaryArchiveWithDescriptor:
            [MTLBinaryArchiveDescriptor new] error:&error];
        if (!archive || ![archive addComputePipelineFunctionsWithDescriptor:pd error:&error])
            return fail("archive carrier", error);
        if (![archive serializeToURL:[NSURL fileURLWithPath:@(archive_path)] error:&error])
            return fail("serialize carrier", error);
        return 0; // No command buffer is created or submitted.
    }
}
void *am_pipeline(void *opaque, const char *library_path, const char *archive_path) {
    @autoreleasepool {
        AFMetal *ctx = (__bridge AFMetal *)opaque;
        NSError *error = nil;
        id<MTLLibrary> library = [ctx.device newLibraryWithURL:
            [NSURL fileURLWithPath:@(library_path)] error:&error];
        if (!library) { fail("pipeline library", error); return NULL; }
        MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
        ad.url = [NSURL fileURLWithPath:@(archive_path)];
        id<MTLBinaryArchive> archive = [ctx.device newBinaryArchiveWithDescriptor:ad error:&error];
        if (!archive) { fail("authored archive", error); return NULL; }
        MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
        pd.computeFunction = [library newFunctionWithName:@"carrier"];
        pd.binaryArchives = @[archive];
        id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithDescriptor:pd
            options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
        if (!pipeline || pipeline.threadExecutionWidth != 32) {
            fail("authored pipeline / archive miss", error); return NULL;
        }
        [ctx.resources addObject:pipeline];
        return (__bridge void *)pipeline;
    }
}
int am_submit(void *opaque, void *pipeline_ptr, void **buffers, uint32_t count,
              uint32_t offset, uint32_t global_x, uint32_t local_x, uint32_t timeout_ms) {
    @autoreleasepool {
        AFMetal *ctx = (__bridge AFMetal *)opaque;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;
        if (ctx.stopped || !pipeline || !buffers || count != 8 || !global_x || !local_x ||
            local_x % 32 || global_x % local_x || local_x > pipeline.maxTotalThreadsPerThreadgroup ||
            !timeout_ms || timeout_ms > 60000) return fail("launch contract", nil);
        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return fail("command buffer / encoder", nil);
        [enc setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < count; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
            if (!buffer || offset >= buffer.length) {
                [enc endEncoding]; return fail("buffer binding", nil);
            }
            [enc setBuffer:buffer offset:offset atIndex:i];
        }
        [enc dispatchThreadgroups:MTLSizeMake(global_x / local_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(local_x, 1, 1)];
        [enc endEncoding];
        dispatch_semaphore_t completed = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> result) {
            (void)result; dispatch_semaphore_signal(completed);
        }];
        ctx.active = cb;
        [cb commit];
        if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW,
                (int64_t)timeout_ms * 1000000))) {
            ctx.stopped = YES; return fail("completion timeout; sequence stopped", nil);
        }
        if (cb.status != MTLCommandBufferStatusCompleted) {
            ctx.stopped = YES; return fail("command buffer failed; sequence stopped", cb.error);
        }
        ctx.active = nil;
        return 0;
    }
}
