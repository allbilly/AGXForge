/* Base M1 / macOS 26A434 direct IOGPU transport. No Metal linkage. */
#include <IOKit/IOKitLib.h>
#include <IOKit/IODataQueueClient.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct an_context { io_connect_t connection; int stopped; };
static _Thread_local char last_error[1024];
static int fail(const char *phase, uint32_t status) {
    snprintf(last_error, sizeof last_error, "%s: 0x%x", phase, status); return -1;
}
const char *an_error(void) { return last_error; }
int an_no_agx_metal(void) {
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const char *name = _dyld_get_image_name(i);
        if (name && strstr(name, "AGXMetal")) return 0;
    }
    return 1;
}
void *an_open(void) {
    if (!an_no_agx_metal()) { fail("AGXMetal driver image loaded in native executor", 0); return NULL; }
    const char *names[] = {"AGXAcceleratorG13G_B0", "AGXAcceleratorG13G"};
    io_service_t service = 0;
    for (unsigned i = 0; i < 2 && !service; i++)
        service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching(names[i]));
    if (!service) { fail("base M1 G13G accelerator not found", 0); return NULL; }
    struct an_context *ctx = calloc(1, sizeof *ctx);
    if (!ctx) { IOObjectRelease(service); fail("context allocation", 0); return NULL; }
    kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0x100005, &ctx->connection);
    IOObjectRelease(service);
    if (kr) { free(ctx); fail("IOServiceOpen", kr); return NULL; }
    return ctx;
}
void an_close(struct an_context *ctx) {
    if (ctx) { IOServiceClose(ctx->connection); free(ctx); }
}
int an_call(struct an_context *ctx, uint32_t selector, const uint64_t *scalars,
            uint32_t scalar_count, const void *input, uint32_t input_size,
            void *output, uint32_t output_size) {
    if (!ctx || ctx->stopped) return fail("closed/stopped native context", 0);
    size_t size = output_size;
    kern_return_t kr = IOConnectCallMethod(ctx->connection, selector, scalars, scalar_count,
        input, input_size, NULL, NULL, output, &size);
    if (kr || size != output_size) { ctx->stopped = 1; return fail("IOConnectCallMethod or output size", kr); }
    return 0;
}
int an_submit(struct an_context *ctx, const void *record, uint32_t bytes) {
    if (!ctx || ctx->stopped || !record || bytes != 64 || !an_no_agx_metal()) return fail("native submit contract", 0);
    kern_return_t kr = IOConnectTrap4(ctx->connection, 0, 1, bytes, (uintptr_t)record, 0);
    if (kr) { ctx->stopped = 1; return fail("IOConnectTrap4", kr); }
    return 0;
}
static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
int an_wait(struct an_context *ctx, IODataQueueMemory *ring, uint64_t token0,
            uint64_t token1, void *receipt, uint32_t timeout_ms) {
    if (!ctx || ctx->stopped || !ring || !token0 || !token1 || token0 == token1 ||
        !receipt || !timeout_ms || timeout_ms > 60000 || !ring->queueSize || ring->queueSize > 16384 - 12)
        return fail("notification wait contract", 0);
    uint64_t deadline = now_ns() + (uint64_t)timeout_ms * 1000000;
    unsigned seen = 0;
    while (now_ns() < deadline) {
        /* A position exactly at queueSize is valid; IODataQueueDequeue wraps
         * it before reading the next entry (IOKitUser IODataQueueClient.c). */
        if (ring->head > ring->queueSize || ring->tail > ring->queueSize) {
            ctx->stopped = 1; return fail("notification ring indices outside mapped page", 0);
        }
        if (!IODataQueueDataAvailable(ring)) {
            struct timespec pause = {.tv_nsec = 100000}; nanosleep(&pause, NULL); continue;
        }
        uint64_t message[5] = {0}; uint32_t size = sizeof message;
        kern_return_t kr = IODataQueueDequeue(ring, message, &size);
        unsigned bit = message[0] == token0 ? 1 : message[0] == token1 ? 2 : 0;
        memcpy((uint8_t *)receipt + (bit == 2 ? 40 : 0), message, 40);
        /* The observed 26A434 completion profile is 40 bytes: callback cookie,
         * start/end timestamps, and two zero status/reserved words. Any other
         * message is refused rather than interpreted as a completion. */
        if (kr || size != sizeof message || !bit || (seen & bit) || !message[1] ||
            message[2] < message[1] || message[3] || message[4]) {
            ctx->stopped = 1;
            snprintf(last_error, sizeof last_error,
                "unexpected/error IOGPU completion: kr=0x%x size=%u cookie=0x%llx bit=%u seen=%u "
                "start=%llu end=%llu status=0x%llx reserved=0x%llx",
                kr, size, (unsigned long long)message[0], bit, seen,
                (unsigned long long)message[1], (unsigned long long)message[2],
                (unsigned long long)message[3], (unsigned long long)message[4]);
            return -1;
        }
        memcpy((uint8_t *)receipt + (bit == 1 ? 0 : 40), message, 40);
        seen |= bit;
        if (seen == 3) return 0;
    }
    ctx->stopped = 1; return fail("IOGPU completion timeout; sequence stopped", 0);
}
