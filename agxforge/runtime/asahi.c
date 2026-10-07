/* SPDX-License-Identifier: MIT */
/* Native Asahi UAPI resource ownership, built against the installed headers. */
#define _GNU_SOURCE
#include "asahi.h"
#include "asahi_internal.h"
#include <drm/asahi_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

_Static_assert(sizeof(uintptr_t) == 8, "64-bit userspace required");
_Static_assert(sizeof(struct drm_asahi_cmd_header) == 8, "UAPI command header");
_Static_assert(sizeof(struct drm_asahi_cmd_compute) == 64, "UAPI compute payload");
_Static_assert(sizeof(struct drm_asahi_gem_bind_op) == 32, "UAPI bind stride");
_Static_assert(sizeof(struct drm_asahi_submit) == 40, "UAPI submit");
_Static_assert(offsetof(struct drm_asahi_submit, queue_id) == 20, "UAPI queue offset");

static char open_error[256];

int af_fail(struct af_device *d, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(d ? d->error : open_error, 256, fmt, args);
    va_end(args);
    return -1;
}

int af_ioctl(struct af_device *d, unsigned long request, void *arg, const char *name)
{
    int ret;
    do { ret = ioctl(d->fd, request, arg); } while (ret < 0 && errno == EINTR);
    if (ret < 0) return af_fail(d, "%s: %s", name, strerror(errno));
    return 0;
}

const char *af_error(struct af_device *d) { return d ? d->error : open_error; }
const char *af_node(struct af_device *d) { return d->node; }

struct af_device *af_open(const char *node, uint32_t slot)
{
    open_error[0] = 0;
    if (slot > 15) { af_fail(NULL, "VA slot must be 0..15"); return NULL; }
    struct af_device *d = calloc(1, sizeof(*d));
    if (!d) { af_fail(NULL, "allocating device: %s", strerror(errno)); return NULL; }
    d->fd = -1;
    glob_t candidates = {0};
    if (!node && glob("/dev/dri/renderD*", 0, NULL, &candidates)) {
        af_fail(NULL, "no DRM render nodes"); free(d); return NULL;
    }
    size_t count = node ? 1 : candidates.gl_pathc;
    for (size_t i = 0; i < count; ++i) {
        const char *path = node ? node : candidates.gl_pathv[i];
        d->fd = open(path, O_RDWR | O_CLOEXEC);
        if (d->fd < 0) continue;
        char name[128] = {0};
        struct drm_version version = {.name_len = sizeof(name) - 1, .name = name};
        if (af_ioctl(d, DRM_IOCTL_VERSION, &version, "DRM_VERSION") || strcmp(name, "asahi")) {
            close(d->fd); d->fd = -1; continue;
        }
        struct drm_asahi_get_params query = {
            .pointer = (uintptr_t)&d->params, .size = sizeof(d->params)
        };
        if (af_ioctl(d, DRM_IOCTL_ASAHI_GET_PARAMS, &query, "GET_PARAMS")) {
            close(d->fd); d->fd = -1; continue;
        }
        snprintf(d->node, sizeof(d->node), "%s", path);
        d->major = version.version_major; d->minor = version.version_minor;
        d->patch = version.version_patchlevel;
        break;
    }
    globfree(&candidates);
    if (d->fd < 0) { af_fail(NULL, "no usable Asahi render node: %s", d->error); free(d); return NULL; }
    struct drm_asahi_params_global *p = &d->params;
    if (p->gpu_generation != 13 || p->gpu_variant != 'G' || p->chip_id != 0x8103 ||
        (p->gpu_revision != 0 && p->gpu_revision != 0x11)) {
        af_fail(NULL, "unsupported profile G%u%c rev 0x%x chip 0x%x; expected base M1 G13G A0/B1",
                p->gpu_generation, p->gpu_variant, p->gpu_revision, p->chip_id);
        close(d->fd); free(d); return NULL;
    }
    uint64_t kernel_size = p->vm_kernel_min_size > (32ull << 30) ? p->vm_kernel_min_size : (32ull << 30);
    uint64_t start = p->vm_start > (1ull << 36) ? p->vm_start : (1ull << 36);
    d->usc_base = AF_ALIGN(start, 1ull << 32) + ((uint64_t)slot << 32);
    d->kernel_start = p->vm_end > kernel_size ? (p->vm_end - kernel_size) & ~(AF_PAGE - 1) : 0;
    d->code_next = d->usc_base + AF_PAGE;
    d->data_next = d->usc_base + (1ull << 32) + ((uint64_t)slot * 0x1000000);
    if (sysconf(_SC_PAGESIZE) != AF_PAGE || !p->max_commands_per_submission ||
        p->vm_end % AF_PAGE || d->data_next >= d->kernel_start) {
        af_fail(NULL, "unsupported page size or GPU VA window"); close(d->fd); free(d); return NULL;
    }
    return d;
}

static int setup(struct af_device *d)
{
    if (d->vm_live) return d->queue_live ? 0 : af_fail(d, "previous queue setup failed");
    struct drm_asahi_vm_create vm = {.kernel_start = d->kernel_start, .kernel_end = d->params.vm_end};
    if (af_ioctl(d, DRM_IOCTL_ASAHI_VM_CREATE, &vm, "VM_CREATE")) return -1;
    d->vm_id = vm.vm_id; d->vm_live = true;
    struct drm_asahi_queue_create queue = {
        .vm_id = vm.vm_id, .priority = DRM_ASAHI_PRIORITY_LOW, .usc_exec_base = d->usc_base
    };
    if (af_ioctl(d, DRM_IOCTL_ASAHI_QUEUE_CREATE, &queue, "QUEUE_CREATE")) return -1;
    d->queue_id = queue.queue_id; d->queue_live = true;
    return 0;
}

struct af_bo *af_alloc(struct af_device *d, uint64_t size, uint32_t access)
{
    if (!size || size > (1ull << 32) || (access & ~7u) || !(access & (AF_READ | AF_WRITE))) {
        af_fail(d, "invalid BO size/access"); return NULL;
    }
    if (d->poisoned || setup(d)) { if (d->poisoned) af_fail(d, "device stopped after failed completion"); return NULL; }
    size = AF_ALIGN(size, AF_PAGE);
    uint64_t *cursor = access & AF_EXEC ? &d->code_next : &d->data_next;
    uint64_t limit = access & AF_EXEC ? d->usc_base + (1ull << 32) : d->kernel_start;
    if (*cursor >= limit || size > limit - *cursor) { af_fail(d, "GPU VA space exhausted"); return NULL; }
    struct af_bo *bo = calloc(1, sizeof(*bo));
    if (!bo) { af_fail(d, "allocating BO: %s", strerror(errno)); return NULL; }
    bo->dev = d; bo->size = size; bo->address = *cursor; bo->access = access;
    bo->next = d->bos; d->bos = bo;
    struct drm_asahi_gem_create gem = {.size = size, .flags = DRM_ASAHI_GEM_VM_PRIVATE, .vm_id = d->vm_id};
    if (af_ioctl(d, DRM_IOCTL_ASAHI_GEM_CREATE, &gem, "GEM_CREATE")) goto fail;
    bo->handle = gem.handle;
    struct drm_asahi_gem_mmap_offset offset = {.handle = bo->handle};
    if (af_ioctl(d, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &offset, "GEM_MMAP_OFFSET")) goto fail;
    bo->cpu = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, d->fd, offset.offset);
    if (bo->cpu == MAP_FAILED) { bo->cpu = NULL; af_fail(d, "mmap: %s", strerror(errno)); goto fail; }
    struct drm_asahi_gem_bind_op op = {
        .flags = ((access & AF_READ) ? DRM_ASAHI_BIND_READ : 0) |
                 ((access & AF_WRITE) ? DRM_ASAHI_BIND_WRITE : 0),
        .handle = bo->handle, .range = size, .addr = bo->address
    };
    struct drm_asahi_vm_bind bind = {.vm_id = d->vm_id, .num_binds = 1, .stride = sizeof(op), .userptr = (uintptr_t)&op};
    if (af_ioctl(d, DRM_IOCTL_ASAHI_VM_BIND, &bind, "VM_BIND")) goto fail;
    bo->bound = true;
    *cursor += size + AF_PAGE; /* unmapped page between objects */
    return bo;
fail: {
    char error[256]; memcpy(error, d->error, sizeof(error));
    af_free(bo); memcpy(d->error, error, sizeof(error)); return NULL;
    }
}

int af_free(struct af_bo *bo)
{
    if (!bo) return 0;
    struct af_device *d = bo->dev;
    if (d->poisoned) return af_fail(d, "failed completion: close the entire device without explicit unbinding");
    int ret = 0;
    if (bo->bound) {
        struct drm_asahi_gem_bind_op op = {.flags = DRM_ASAHI_BIND_UNBIND, .range = bo->size, .addr = bo->address};
        struct drm_asahi_vm_bind bind = {.vm_id = d->vm_id, .num_binds = 1, .stride = sizeof(op), .userptr = (uintptr_t)&op};
        if (af_ioctl(d, DRM_IOCTL_ASAHI_VM_BIND, &bind, "VM_UNBIND")) ret = -1;
    }
    if (bo->cpu && munmap(bo->cpu, bo->size)) { af_fail(d, "munmap: %s", strerror(errno)); ret = -1; }
    if (bo->handle) {
        struct drm_gem_close close_bo = {.handle = bo->handle};
        if (af_ioctl(d, DRM_IOCTL_GEM_CLOSE, &close_bo, "GEM_CLOSE")) ret = -1;
    }
    struct af_bo **link = &d->bos;
    while (*link && *link != bo) link = &(*link)->next;
    if (*link) *link = bo->next;
    free(bo);
    return ret;
}

int af_close(struct af_device *d)
{
    if (!d) return 0;
    int ret = 0;
    if (d->poisoned) {
        /* A timeout does not establish completion or recovery. Let DRM file
         * teardown and the driver's job/VM references own pending resources.
         * Do not explicitly unbind objects which may still be executing. */
        if (close(d->fd)) ret = -1;
        while (d->bos) {
            struct af_bo *bo = d->bos;
            d->bos = bo->next;
            if (bo->cpu && munmap(bo->cpu, bo->size)) ret = -1;
            free(bo);
        }
        if (ret) snprintf(open_error, sizeof(open_error), "failed-completion teardown: %s", strerror(errno));
        free(d);
        return ret;
    }
    if (d->sync_live) {
        struct drm_syncobj_destroy sync = {.handle = d->sync_id};
        if (af_ioctl(d, DRM_IOCTL_SYNCOBJ_DESTROY, &sync, "SYNCOBJ_DESTROY")) ret = -1;
    }
    /* Every admitted dispatch has completed before ordinary resource teardown. */
    if (d->queue_live) {
        struct drm_asahi_queue_destroy queue = {.queue_id = d->queue_id};
        if (af_ioctl(d, DRM_IOCTL_ASAHI_QUEUE_DESTROY, &queue, "QUEUE_DESTROY")) ret = -1;
    }
    while (d->bos) if (af_free(d->bos)) ret = -1;
    if (d->vm_live) {
        struct drm_asahi_vm_destroy vm = {.vm_id = d->vm_id};
        if (af_ioctl(d, DRM_IOCTL_ASAHI_VM_DESTROY, &vm, "VM_DESTROY")) ret = -1;
    }
    if (close(d->fd)) { af_fail(d, "close: %s", strerror(errno)); ret = -1; }
    if (ret) snprintf(open_error, sizeof(open_error), "%s", d->error);
    free(d);
    return ret;
}

void *af_map(struct af_bo *bo) { return bo->cpu; }
uint64_t af_address(struct af_bo *bo) { return bo->address; }
uint64_t af_size(struct af_bo *bo) { return bo->size; }
uint32_t af_handle(struct af_bo *bo) { return bo->handle; }

uint64_t af_param(struct af_device *d, uint32_t key)
{
    struct drm_asahi_params_global *p = &d->params;
    if (key >= AF_PARAM_CORE_MASK && key < AF_PARAM_CORE_MASK + DRM_ASAHI_MAX_CLUSTERS)
        return p->core_masks[key - AF_PARAM_CORE_MASK];
    const uint64_t values[] = {
        p->features, p->gpu_generation, p->gpu_variant, p->gpu_revision, p->chip_id,
        p->vm_start, p->vm_end, p->vm_kernel_min_size, p->max_commands_per_submission,
        p->max_attachments, p->command_timestamp_frequency_hz, p->num_dies, p->num_clusters_total,
        p->num_cores_per_cluster, p->max_frequency_khz, d->major, d->minor, d->patch,
        d->usc_base, d->kernel_start, d->vm_id, d->queue_id
    };
    return key < sizeof(values) / sizeof(values[0]) ? values[key] : 0;
}
