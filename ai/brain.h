#ifndef __BRAIN_H_DEF__
#define __BRAIN_H_DEF__

#include <stdint.h>
#include <stddef.h>

void brain_init();
bool brain_enabled();
bool brain_headless();
void brain_begin_rollout();
uint32_t brain_num_rollouts();
uint8_t brain_controller_bits();
bool brain_on_frame(float *frame_reward, int *action_idx);
void brain_bind_nes(const uint8_t *ram, const uint32_t *screen);

const uint32_t *brain_get_screen(uint32_t &w, uint32_t &h);

#endif // __BRAIN_H_DEF__


