#include "hqn.h"
#include "hqn_gui_controller.h"

#include "gif.h"

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

static int frame = 0;
static const unsigned int crc32_table[256] =
{
  0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9,
  0x130476dc, 0x17c56b6b, 0x1a864db2, 0x1e475005,
  0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61,
  0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd,
  0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9,
  0x5f15adac, 0x5bd4b01b, 0x569796c2, 0x52568b75,
  0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011,
  0x791d4014, 0x7ddc5da3, 0x709f7b7a, 0x745e66cd,
  0x9823b6e0, 0x9ce2ab57, 0x91a18d8e, 0x95609039,
  0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5,
  0xbe2b5b58, 0xbaea46ef, 0xb7a96036, 0xb3687d81,
  0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d,
  0xd4326d90, 0xd0f37027, 0xddb056fe, 0xd9714b49,
  0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
  0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1,
  0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a, 0xec7dd02d,
  0x34867077, 0x30476dc0, 0x3d044b19, 0x39c556ae,
  0x278206ab, 0x23431b1c, 0x2e003dc5, 0x2ac12072,
  0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16,
  0x018aeb13, 0x054bf6a4, 0x0808d07d, 0x0cc9cdca,
  0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
  0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02,
  0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1, 0x53dc6066,
  0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba,
  0xaca5c697, 0xa864db20, 0xa527fdf9, 0xa1e6e04e,
  0xbfa1b04b, 0xbb60adfc, 0xb6238b25, 0xb2e29692,
  0x8aad2b2f, 0x8e6c3698, 0x832f1041, 0x87ee0df6,
  0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a,
  0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e,
  0xf3b06b3b, 0xf771768c, 0xfa325055, 0xfef34de2,
  0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686,
  0xd5b88683, 0xd1799b34, 0xdc3abded, 0xd8fba05a,
  0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637,
  0x7a089632, 0x7ec98b85, 0x738aad5c, 0x774bb0eb,
  0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f,
  0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53,
  0x251d3b9e, 0x21dc2629, 0x2c9f00f0, 0x285e1d47,
  0x36194d42, 0x32d850f5, 0x3f9b762c, 0x3b5a6b9b,
  0x0315d626, 0x07d4cb91, 0x0a97ed48, 0x0e56f0ff,
  0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623,
  0xf12f560e, 0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7,
  0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
  0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f,
  0xc423cd6a, 0xc0e2d0dd, 0xcda1f604, 0xc960ebb3,
  0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7,
  0xae3afba2, 0xaafbe615, 0xa7b8c0cc, 0xa379dd7b,
  0x9b3660c6, 0x9ff77d71, 0x92b45ba8, 0x9675461f,
  0x8832161a, 0x8cf30bad, 0x81b02d74, 0x857130c3,
  0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
  0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c,
  0x7b827d21, 0x7f436096, 0x7200464f, 0x76c15bf8,
  0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24,
  0x119b4be9, 0x155a565e, 0x18197087, 0x1cd86d30,
  0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec,
  0x3793a651, 0x3352bbe6, 0x3e119d3f, 0x3ad08088,
  0x2497d08d, 0x2056cd3a, 0x2d15ebe3, 0x29d4f654,
  0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0,
  0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb, 0xdbee767c,
  0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18,
  0xf0a5bd1d, 0xf464a0aa, 0xf9278673, 0xfde69bc4,
  0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0,
  0x9abc8bd5, 0x9e7d9662, 0x933eb0bb, 0x97ffad0c,
  0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
  0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4
};

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
		std::array<int32_t, 32*30> mini_screen;
		//for(int y = 0; y < 240 / 8; ++y)
		//{
		//	for(int x = 0; x < 256 / 8; ++x)
		//	{
		//		uint32_t hash = 0xFFFFFFFF;
		//		for(int i = 0; i < 8; ++i)
		//		{
		//			for(int j = 0; j < 8; ++j)
		//			{
		//				uint32_t idx = (y * 8 + j) + x * 8 + i;
		//				hash = (hash << 8) ^ crc32_table[((hash >> 24) ^ frame_pixels[idx]) & 255];
		//			}
		//		}
		//		mini_screen[y * 32 + x] = frame_pixels[y * 8 + x * 8];//static_cast<float>(hash) / static_cast<float>(0xFFFFFFFF);
		//	}
		//}
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
			//for(int y = 0; y < 240 / 8; ++y)
			//{
			//	for(int x = 0; x < 256 / 8; ++x)
			//	{
			//		frame_pixels[y * 256 + x] = static_cast<int>(mini_screen[y * 32 + x]);
			//	}
			//}
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
		brain_bind_cpu_mem(hqn_state.emu()->low_mem());

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
