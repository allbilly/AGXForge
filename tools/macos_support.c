/* Local driver support-code collector. No captured bytes are distributed.
 * Preparation only: this interposer refuses every GPU submission.
 * SPDX-License-Identifier: MIT
 */
#include <IOKit/IOKitLib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const void *code_heap;
static size_t code_bytes;
static unsigned submissions;

static kern_return_t collect_call(mach_port_t connection, uint32_t selector,
    const uint64_t *input, uint32_t input_count, const void *input_struct,
    size_t input_size, uint64_t *output, uint32_t *output_count,
    void *output_struct, size_t *output_size) {
    kern_return_t status = IOConnectCallMethod(connection, selector, input,
        input_count, input_struct, input_size, output, output_count,
        output_struct, output_size);
    if (status == KERN_SUCCESS && selector == 9 && output_struct &&
        output_size && *output_size >= 48 && input_struct && input_size == 104) {
        uint64_t gpu, cpu, size;
        uint32_t flags;
        memcpy(&gpu, output_struct, 8);
        memcpy(&cpu, (const uint8_t *)output_struct + 8, 8);
        memcpy(&size, (const uint8_t *)output_struct + 40, 8);
        memcpy(&flags, (const uint8_t *)input_struct + 20, 4);
        if (gpu == 0 && cpu && size == 65536 && flags == 0x8430) {
            code_heap = (const void *)(uintptr_t)cpu;
            code_bytes = (size_t)size;
        }
    }
    return status;
}

static kern_return_t refuse_submit(mach_port_t connection, uint32_t index,
    uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d) {
    (void)connection; (void)index; (void)a; (void)b; (void)c; (void)d;
    submissions++;
    return kIOReturnNotPermitted;
}

__attribute__((visibility("default")))
int agxforge_write_support(const char *path) {
    if (!path || !code_heap || code_bytes != 65536 || submissions) return -1;
    FILE *file = fopen(path, "wb");
    if (!file) return -2;
    size_t written = fwrite(code_heap, 1, 0x5000, file);
    int status = fclose(file);
    return written == 0x5000 && status == 0 ? 0 : -3;
}

#define INTERPOSE(replacement, original) \
    __attribute__((used)) static const struct { const void *new_fn; const void *old_fn; } \
    interpose_##replacement __attribute__((section("__DATA,__interpose"))) = \
    { (const void *)&replacement, (const void *)&original }
INTERPOSE(collect_call, IOConnectCallMethod);
INTERPOSE(refuse_submit, IOConnectTrap4);
