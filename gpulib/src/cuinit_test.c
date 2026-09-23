/* cuinit_test.c -- minimal CUDA driver-API probe via dlopen (no CUDA headers
 * needed).  Prints cuInit rc, device count, and the first device name. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDriverGetVersion_t)(int *);
typedef int (*cuDeviceGetCount_t)(int *);
typedef int (*cuDeviceGetName_t)(char *, int, int);
typedef int (*cuGetErrorString_t)(int, const char **);
int main(void)
{
    const char *lib = getenv("CUDA_LIB") ? getenv("CUDA_LIB") : "libcuda.so.1";
    void *h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen(%s) failed: %s\n", lib, dlerror()); return 1; }
    cuInit_t cuInit = (cuInit_t)dlsym(h, "cuInit");
    cuDriverGetVersion_t cuDriverGetVersion = (cuDriverGetVersion_t)dlsym(h, "cuDriverGetVersion");
    cuDeviceGetCount_t cuDeviceGetCount = (cuDeviceGetCount_t)dlsym(h, "cuDeviceGetCount");
    cuDeviceGetName_t cuDeviceGetName = (cuDeviceGetName_t)dlsym(h, "cuDeviceGetName");
    cuGetErrorString_t cuGetErrorString = (cuGetErrorString_t)dlsym(h, "cuGetErrorString");
    int v = 0; cuDriverGetVersion(&v);
    printf("cuDriverGetVersion: %d\n", v);
    int rc = cuInit(0);
    const char *msg = "?"; if (cuGetErrorString) cuGetErrorString(rc, &msg);
    printf("cuInit rc=%d (%s)\n", rc, msg);
    if (rc) return 2;
    int n = -1; rc = cuDeviceGetCount(&n);
    printf("cuDeviceGetCount rc=%d n=%d\n", rc, n);
    if (n > 0) { char name[256]; cuDeviceGetName(name, 256, 0); printf("device 0: %s\n", name); }
    return 0;
}
