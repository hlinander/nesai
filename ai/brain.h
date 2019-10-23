#ifndef __BRAIN_H_DEF__
#define __BRAIN_H_DEF__

#include <stdint.h>
#include <stddef.h>

void brain_init();
bool brain_enabled();
bool brain_headless();
uint8_t brain_controller_bits();
bool brain_on_frame();
void brain_bind_cpu_mem(const uint8_t *ram);

#endif // __BRAIN_H_DEF__

