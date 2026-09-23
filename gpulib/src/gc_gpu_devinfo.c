/*
 * gc_gpu_devinfo.c -- device query for libgcchemgpu.so, via the CUDA driver
 * API loaded with dlopen.
 *
 * Why not the OpenACC API: the model is a gfortran build and carries libgomp,
 * which also exports acc_get_num_devices / acc_init / acc_get_property_string.
 * Those bind to libgomp's own (GPU-less) implementations at runtime and abort
 * with "libgomp: no device found".  The compute kernels are unaffected --
 * nvfortran lowers !$acc regions to private __pgi_uacc_* entry points in
 * libacchost, which libgomp does not define -- so only this query routine
 * needs to avoid the public acc_* API.
 *
 * Exposes: void gc_gpu_info(int *nDev, char *devName, int nameLen)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDeviceGetCount_t)(int *);
typedef int (*cuDeviceGetName_t)(char *, int, int);

void gc_gpu_info(int *nDev, char *devName, int nameLen)
{
    *nDev = 0;
    if (nameLen > 0) devName[0] = '\0';

    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return;

    cuInit_t           cuInit           = (cuInit_t)          dlsym(h, "cuInit");
    cuDeviceGetCount_t cuDeviceGetCount = (cuDeviceGetCount_t)dlsym(h, "cuDeviceGetCount");
    cuDeviceGetName_t  cuDeviceGetName  = (cuDeviceGetName_t) dlsym(h, "cuDeviceGetName");
    if (!cuInit || !cuDeviceGetCount || !cuDeviceGetName) return;

    if (cuInit(0) != 0) return;
    int n = 0;
    if (cuDeviceGetCount(&n) != 0) return;
    *nDev = n;
    if (n > 0 && nameLen > 1) {
        char buf[256];
        buf[0] = '\0';
        if (cuDeviceGetName(buf, (int)sizeof(buf), 0) == 0) {
            int m = (int)strlen(buf);
            if (m > nameLen - 1) m = nameLen - 1;
            memcpy(devName, buf, (size_t)m);
            devName[m] = '\0';
        }
    }
}


/*
 * gc_gpu_set_device(dev) -- make NVIDIA device `dev` the current OpenACC
 * device of the CALLING HOST THREAD, and return the device number the
 * OpenACC runtime reports afterwards (-1 on failure).
 *
 * Why dlsym and not the acc_set_device_num API or the `!$acc set` directive:
 * both resolve the PUBLIC symbol acc_set_device_num, and in the gfortran-
 * built model that binds to libgomp's implementation ("libgomp: no device
 * found").  Looking the symbol up on the handle of NVIDIA's own runtime
 * library bypasses the global symbol scope and reaches the real thing.
 */
typedef void (*acc_set_device_num_t)(int, int);
typedef int  (*acc_get_device_num_t)(int);
#define GC_ACC_DEVICE_NVIDIA 4   /* nvhpc openacc.h: acc_device_nvidia = 4 */
int gc_gpu_set_device(int dev)
{
    static void *h = 0;
    static acc_set_device_num_t setnum = 0;
    static acc_get_device_num_t getnum = 0;
    if (!h) {
        const char *libs[] = { "libacchost.so", "libaccdevice.so", "libaccdevaux.so", 0 };
        for (int i = 0; libs[i] && !setnum; i++) {
            h = dlopen(libs[i], RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
            if (!h) h = dlopen(libs[i], RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            setnum = (acc_set_device_num_t)dlsym(h, "acc_set_device_num");
            getnum = (acc_get_device_num_t)dlsym(h, "acc_get_device_num");
            if (!setnum) { h = 0; }
        }
    }
    if (!setnum) return -1;
    setnum(dev, GC_ACC_DEVICE_NVIDIA);
    return getnum ? getnum(GC_ACC_DEVICE_NVIDIA) : dev;
}
