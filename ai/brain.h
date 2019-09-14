#ifndef __BRAIN_H_DEF__
#define __BRAIN_H_DEF__

#include <stdint.h>
#include <stddef.h>

void brain_init();
bool brain_enabled();
bool brain_headless();
bool brain_continue();
uint8_t brain_controller_bits();
void brain_on_frame(const uint8_t *ram, size_t n);

#endif // __BRAIN_H_DEF__


