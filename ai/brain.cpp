#undef OPENGL
#include "brain.h"
#include "model.h"

#include <stdlib.h>
#include <iostream>

static uint8_t gp_strobe = 0; 
static uint8_t gp_bits = 0;

static Model model{0.001};

static bool enabled = false;
static bool headless = false;
static bool running = false;
static bool show_fps = false;
static uint32_t fps = 0;
static uint64_t next_fps = 0;

static bool bool_env(const char *env)
{
	const char * const s = getenv(env);
	if(!s)
	{
		return false;
	}
	return (0 != strcmp(s, "0"));
}

static uint64_t get_ms() 
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);

	uint64_t ms  = ts.tv_nsec / 1000000;
	ms += ts.tv_sec * 1000;
	return ms;
}

void brain_init()
{
	enabled = bool_env("BE");
	headless = bool_env("HL");
	show_fps = bool_env("FPS");
	running = true;
}

bool brain_enabled()
{
	return enabled;
}

bool brain_headless()
{
	return enabled && headless;
}

bool brain_continue()
{
	return enabled && running;
}

uint8_t brain_controller_bits()
{
	return gp_bits;
}

void brain_on_frame(const uint8_t *ram, size_t n)
{
	if(n != STATE_SIZE)
	{
		std::cout << "BAD RAM SIZE (on frame): " << n << " vs. " << STATE_SIZE << std::endl;
		exit(1);
	}

	StateType s;
	for(size_t i = 0; i < STATE_SIZE; ++i) {
		s[i] = static_cast<float>(ram[i]) / 255.0;
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

	++fps;
	auto now = get_ms();

	if(next_fps < now)
	{
		std::cout << "FPS: " << fps << std::endl;
		fps = 0;
		next_fps = now + 1000;
	}
}

/*
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
*/