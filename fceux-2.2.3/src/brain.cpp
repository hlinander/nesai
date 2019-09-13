#include "brain.h"

#include <stdlib.h>

static uint8 gp_strobe = 0; 
static uint8 gp_bits = 0;

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