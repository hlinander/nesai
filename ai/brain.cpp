#undef OPENGL
#include "brain.h"
#include "model.h"

#include <stdlib.h>

static uint8 gp_strobe = 0; 
static uint8 gp_bits = 0;

Model model{0.001};

bool brain_enabled()
{
	return true;
}

bool brain_headless()
{
	return true;
}

extern uint8 *RAM;
void brain_on_frame()
{
	StateType s;
	for(size_t i = 0; i < STATE_SIZE; ++i) {
		s[i] = static_cast<float>(RAM[i]) / 255.0;
	}	
	ActionType a = model.get_action(s);
	gp_bits = 0;
	gp_bits |= a[static_cast<size_t>(Action::A)] << 0;
	gp_bits |= a[static_cast<size_t>(Action::B)] << 1;
	gp_bits |= a[static_cast<size_t>(Action::SELECT)] << 2;
	gp_bits |= a[static_cast<size_t>(Action::START)] << 3;
	gp_bits |= a[static_cast<size_t>(Action::UP)] << 4;
	gp_bits |= a[static_cast<size_t>(Action::DOWN)] << 5;
	gp_bits |= a[static_cast<size_t>(Action::LEFT)] << 6;
	gp_bits |= a[static_cast<size_t>(Action::RIGHT)] << 7;
}

static uint8 input_read(int w)
{
	if(0 != w)
	{
		return 0;
	}

	static int count = 0;
 	if(0 == (count++ & 0x7F))
	{
		gp_bits = (count & 0x80) ? 8 : 0;
	}
	if(gp_strobe >= 8)
	{
		printf("NES game not strobing correctly\n");
		return 0; // bug
	}
	uint8 ret = (gp_bits >> gp_strobe) & 1;
	++gp_strobe;
	return ret;
}

static void input_strobe(int w)
{
	gp_strobe = 0;
}

static void input_update(int w, void *data, int arg)
{
	(void)w;
	(void)data;
	(void)arg;
	// TODO is this function important?
}

static void input_log(int w, MovieRecord* mr)
{
	(void)w;
	(void)mr;
}

static void input_load(int w, MovieRecord* mr)
{
	(void)w;
	(void)mr;
}

INPUTC brain_input = {
	input_read,
	0,
	input_strobe,
	input_update,
	0,
	0,
	input_log,
	input_load
};