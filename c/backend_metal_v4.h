#ifndef COLI_BACKEND_METAL_V4_H
#define COLI_BACKEND_METAL_V4_H

#include "backend_metal_v4_seam.h"

#if defined(COLI_METAL)
#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_metal_init(const char *metallib_path);
void coli_v4_metal_shutdown(void);
int coli_v4_metal_available(void);
const char *coli_v4_metal_library_kind(void);

#ifdef __cplusplus
}
#endif
#endif

#endif
