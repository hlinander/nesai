#include "hqn.h"
#include "hqn_gui_controller.h"

#include "gif.h"
#include "font.h"

#include <brain.h>

#include <string>
#include <iostream>
#include <SDL.h>
#include <SDL_video.h>
#include <SDL_keyboard.h>
#include <SDL_scancode.h>
#include <SDL_mouse.h>
#include <array>
#include <fstream>

using namespace hqn;

static HQNState hqn_state;

static bool is_human = false;

static const char *gif = nullptr;
static GifWriter g;

static std::array<uint32_t, 32*30> mini_screen;

static int frame = 0;

#define GET_GUI() hqn::GUIController *gui = static_cast<hqn::GUIController*>(hqn_state.getListener())

static void ngui_setscale(int scale = 1)
{
	GET_GUI();
	gui->setScale(scale);
}

static bool ngui_enable()
{
    if(!SDL_WasInit(SDL_INIT_VIDEO))
    {
        if(SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
        {
            return false;
        }
    }

    GET_GUI();
    if(!gui)
    {
        hqn::GUIController *controller = nullptr;

        if(controller = hqn::GUIController::create(hqn_state))
        {
            hqn_state.setListener(controller);
            controller->setCloseOperation(hqn::GUIController::CLOSE_DELETE);
        }
        else
        {
            return false;
        }
    }
    return true;
}

static void nemu_setframerate(int fps)
{
	hqn_state.setFramerate(fps);
}

static bool nemu_loadrom(const char *romname)
{
	const char *err = hqn_state.loadROM(romname);
	if(err)
	{
		std::cout << "Error when loading ROM " << romname << ": " << err << std::endl;
		return false;
	}
	return true;
}

static void njoypad_set(int pad, uint8_t bits)
{
	hqn_state.joypad[pad] = bits;
}

static void nemu_frameadvance()
{
	hqn_state.advanceFrame();
}

static void nemu_hard_reset()
{
	hqn_state.reset(true);
}

static bool ngui_isenabled()
{
    GET_GUI();
    return (nullptr != gui);
}

struct gamepad_binding
{
	int sdl_scancode;
	uint8_t bit;
};

static int32_t frame_pixels[256 * 240];

static void *state_data = nullptr;
static size_t state_size = 0;

void load_state() {
	hqn_state.loadState(state_data, state_size);
}

void save_state() {
	size_t size_out;
	size_t new_size;
	hqn_state.saveStateSize(&new_size);
	if(new_size > state_size)
	{
		if(state_data != nullptr) {
			free(state_data);
		}
		state_data = malloc(new_size);
		state_size = new_size;
	}
	hqn_state.saveState(state_data, state_size, &size_out);
}

void save_state_disk() {
	save_state();
	auto sf = std::ofstream("/tmp/save", std::ios::out | std::ios::binary);
	sf.write(state_data, state_size);
}

void load_state_disk() {
	auto sf = std::ifstream("/tmp/save", std::ios::ate | std::ios::binary);
	if(!sf.fail())
	{
		size_t size = sf.tellg();
		sf.seekg(0, std::ios::beg);
		if(state_data != nullptr) {
			free(state_data);
		}
		state_data = malloc(size);
		state_size = size;
		sf.read(state_data, size);
		load_state();
	}
}

static void write_char(int at_x, int at_y, char n, uint32_t fg, uint32_t bg = 0, int scale = 1)
{
	for(int y = at_y; y < (8 * scale); ++y)
	{
		uint8_t mask = 0x80;
		int shift_in = scale;
		for(int x = at_x; x < (8 * scale); ++x)
		{
			if(mask & font_data[n][(y - at_y) / scale])
			{
				if(x < 256 && y < 240)
				{
					frame_pixels[x + y * 256] = fg;
				}
			}
			else
			{
				if(bg && x < 256 && y < 240)
				{
					frame_pixels[x + y * 256] = bg;
				}
			}
			if(0 == --shift_in)
			{
				mask >>= 1;
				shift_in = scale;
			}
		}
	}
}

static void write_string(int at_x, int at_y, const std::string &text, uint32_t fg, uint32_t bg = 0, int scale = 1)
{
	for(const auto &c : text)
	{
		write_char(at_x++, at_y, c, fg, bg, scale);
	}
}

static int run_brain()
{
	gamepad_binding bindings[] = 
	{
		// A
		{ SDL_SCANCODE_X, 0x01 },
		// B
		{ SDL_SCANCODE_Z, 0x02 },
		// Select
		{ SDL_SCANCODE_A, 0x04 },
		// Start
		{ SDL_SCANCODE_S, 0x08 },
		// Up
		{ SDL_SCANCODE_UP, 0x10 },
		// Down
		{ SDL_SCANCODE_DOWN, 0x20 },
		// Left
		{ SDL_SCANCODE_LEFT, 0x40 },
		// RIGHT
		{ SDL_SCANCODE_RIGHT, 0x80 },
	};

	float frame_reward = 0;
	float total_reward = 0;
	if(!getenv("RESET")) 
	{
		load_state_disk();
	}
	while(brain_on_frame(&frame_reward))
	{
		uint8_t bits = brain_controller_bits();

		if(is_human)
		{
			if(!ngui_isenabled())
			{
				break;
			}

			const Uint8 *kb = SDL_GetKeyboardState(NULL);

			bits = 0;

			for(size_t i = 0; i < (sizeof(bindings) / sizeof(bindings[0])); ++i)
			{
				const auto &b = bindings[i];
				if(kb[b.sdl_scancode])
				{
					bits |= b.bit;
				}
			}
		}

		if(!brain_headless())
		{
		}

		njoypad_set(0, bits);
		nemu_frameadvance();
		hqn_state.blit(frame_pixels, HQNState::NES_VIDEO_PALETTE, 0, 0, 0, 0);
		for(int y = 0; y < 240 / 8; ++y)
		{
			for(int x = 0; x < 256 / 8; ++x)
			{
				// uint32_t hash = 0xFFFFFFFF;
				uint32_t r = 0;
				uint32_t g = 0;
				uint32_t b = 0;
				for(int i = 0; i < 8; ++i)
				{
					for(int j = 0; j < 8; ++j)
					{
						uint32_t idx = (y * 8 + j) * 256 + x * 8 + i;
						// hash = (hash << 8) ^ crc32_table[((hash >> 24) ^ frame_pixels[idx]) & 255];
						r += frame_pixels[idx] >> 16 & 0xff;
						g += frame_pixels[idx] >> 8 & 0xff;
						b += frame_pixels[idx] & 0xff;
					}
				}
				// mini_screen[y * 32 + x] = frame_pixels[y * 8 * 256 + x * 8];//static_cast<float>(hash) / static_cast<float>(0xFFFFFFFF);
				mini_screen[y * 32 + x] = ((b >> 6) << 16) | ((g >> 6) << 8) | (r >> 6) ;//static_cast<float>(hash) / static_cast<float>(0xFFFFFFFF);
			}
		}
		frame++;
		if(gif && !(frame&0xf))
		{
			for(int i = 0; i < 256*240; ++i) {
				auto r = frame_pixels[i] >> 16 & 0xff;
				auto g = frame_pixels[i] >> 8 & 0xff;
				auto b = frame_pixels[i] & 0xff;
				frame_pixels[i] = 0xff000000 | (b << 16) | (g << 8) | (r);
			}
			for(int i = 0; i < 3; ++i) {
			for(int j = 0; j < 256; ++j) {
				if(fabs(frame_reward) / 20.0 * 256.0 > static_cast<float>(j))
				{
					if(frame_reward > 0)
					{
						frame_pixels[i * 256 + j] = 0xff000000 | (0 << 16) | (255 << 8) | (0);
					}
					else
					{
						frame_pixels[i * 256 + j] = 0xff000000 | (0 << 16) | (0 << 8) | (255);
					}
				}
			}
			}
			for(int y = 0; y < 240 / 8; ++y)
			{
				for(int x = 0; x < 256 / 8; ++x)
				{
					frame_pixels[(y + 10) * 256 + x] = static_cast<int>(mini_screen[y * 32 + x]);
				}
			}

			total_reward += frame_reward;
			write_string(8, 200, std::to_string(total_reward), 0xFF0000FF, 0, 2);
			GifWriteFrame(&g, reinterpret_cast<uint8_t *>(frame_pixels), 256, 240, 1);
		}
	}
	save_state_disk();
	return 0;
}

static int get_frame_rate()
{
	if(is_human)
	{
		return 60;
	}

	const char *fr = getenv("FPS");
	if(fr)
	{
		return atoi(fr);
	}
	return 0;
}

int main(int argc, const char *argv[])
{
	SDL_SetMainReady();
	brain_init();

	if(!brain_enabled())
	{
		std::cout << "Brain not enabled, use BE=<script.lua>" << std::endl;
		return 1;
	}

	if(argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <rom>" << std::endl;
		return 1;
	}

	gif = getenv("GIF");

	if(gif)
	{
		GifBegin(&g, gif, 256, 240, 1);
	}

	const char * const human = getenv("HUMAN");
	is_human = (human && '1' == *human);

	if(is_human && brain_headless())
	{
		std::cout << "Cant run human mode in headless. Remove HL=1" << std::endl;
		return 1;
	}

	if(!nemu_loadrom(argv[1]))
	{
		std::cout << "No rom :(" << std::endl;
		return 1;
	}

	if(!brain_headless())
	{
		if(!ngui_enable())
		{
			std::cout << "Unable to enable GUI" << std::endl;
			return 1;
		}
		ngui_setscale(2);
	}

	nemu_setframerate(get_frame_rate());

	uint32_t rollout = 0;

	for(;;)
	{
		brain_bind_cpu_mem(hqn_state.emu()->low_mem(), &mini_screen[0]);

		int rc = run_brain();

		if(0 != rc)
		{
			std::cout << "Brain failed to run with: " << rc << std::endl;
			return rc;
		}

		if(++rollout == brain_num_rollouts())
		{
			std::cout << "Ran " << rollout << " rollouts successfully! :)" << std::endl;
			break;
		}

		nemu_hard_reset();
	}

	if(gif)
	{
		GifEnd(&g);
	}

	return 0;
}
