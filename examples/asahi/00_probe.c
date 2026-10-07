/* SPDX-License-Identifier: MIT */
/* No dispatch: exercise VM, queue, BO map/bind/unbind/close lifecycles. */
#include "asahi.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv)
{
    const char *node = NULL;
    unsigned repeat = 100;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) node = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            char *end; unsigned long value = strtoul(argv[++i], &end, 10);
            if (*end || value < 1 || value > 10000) return 2;
            repeat = value;
        } else { fprintf(stderr, "usage: %s [--device NODE] [--repeat 1..10000]\n", argv[0]); return 2; }
    }
    for (unsigned i = 0; i < repeat; ++i) {
        struct af_device *d = af_open(node, i % 16);
        if (!d) { fprintf(stderr, "%s\n", af_error(NULL)); return 1; }
        struct af_bo *bo = af_alloc(d, 16384, AF_READ | AF_WRITE);
        if (!bo) { fprintf(stderr, "%s\n", af_error(d)); af_close(d); return 1; }
        memset(af_map(bo), 0x5a, af_size(bo));
        if (i == 0) printf("{\"node\":\"%s\",\"generation\":%" PRIu64 ",\"variant\":%" PRIu64
            ",\"revision\":%" PRIu64 ",\"chip_id\":%" PRIu64 ",\"features\":%" PRIu64
            ",\"vm_start\":%" PRIu64 ",\"vm_end\":%" PRIu64 ",\"kernel_min\":%" PRIu64 ",",
            af_node(d), af_param(d, AF_GENERATION), af_param(d, AF_VARIANT), af_param(d, AF_REVISION),
            af_param(d, AF_CHIP), af_param(d, AF_FEATURES), af_param(d, AF_VM_START),
            af_param(d, AF_VM_END), af_param(d, AF_KERNEL_MIN));
        int freed = af_free(bo), closed = af_close(d);
        if (freed || closed) { fprintf(stderr, "cleanup failed\n"); return 1; }
    }
    printf("\"status\":\"PASS\",\"gpu_execution\":\"NOT_RUN\",\"resource_cycles\":%u}\n", repeat);
    return 0;
}
