#include "brain.h"

static uint8_t controller = 0;

bool brain_enabled()
{
	return true;
}

bool brain_with_video()
{
	return true;
}

void brain_on_frame()
{
	
}

static uint8 input_read(int w)
{
	return (0 == w) ? controller : 0;
}

static void input_strobe(int w)
{
	// nah
	(void)w;
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