#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdint.h>
#include <cstring>

static uint16_t swap_endian(uint16_t n)
{
	return __builtin_bswap16(n);
}

static uint32_t swap_endian(uint32_t n)
{
	return __builtin_bswap32(n);
}

static uint64_t swap_endian(uint64_t n)
{
	return __builtin_bswap64(n);
}

template <typename T>
static T read_int(std::ifstream &in)
{
	T v;
	if(sizeof(T) != in.read(reinterpret_cast<char *>(&v), sizeof(v)).gcount())
	{
		throw std::runtime_error("Cant read int...");
	}
	return swap_endian(v);
}

static std::string read_str(std::ifstream &in)
{
	auto len = read_int<uint32_t>(in);
	std::string s;
	s.resize(len);
	if(len != in.read(const_cast<char *>(s.data()), len).gcount())
	{
		throw std::runtime_error("Cant read string...");
	}
	return s;
}

static double read_double(std::ifstream &in)
{
	auto v = read_int<uint64_t>(in);
	return *((double *)&v);
}

template <typename T>
static void write_int(std::ofstream &out, T n)
{
	n = swap_endian(n);
	out.write(reinterpret_cast<const char *>(&n), sizeof(n));
}

struct rds_data
{
	enum class rds_type
	{
		object = 0,
		flt,
		flt_arr,
		flt_arr_arr
	};

	rds_data()
		: m_type{rds_type::object}
	{
	}

	rds_data &operator [](const std::string &key)
	{
		m_children.emplace(key, std::make_shared<rds_data>());
		return *m_children[key].get();
	}

	rds_data &operator =(double f)
	{
		*((double *)&m_data.v) = f;
		m_type = rds_type::flt;
		return *this;
	}

	rds_data &operator =(float f)
	{
		*((double *)&m_data.v) = (double)f;
		m_type = rds_type::flt;
		return *this;
	}

	rds_data &operator =(const std::vector<float> *p)
	{
		m_data.ptr = (void *)p;
		m_type = rds_type::flt_arr;
		return *this;
	}

	rds_data &operator =(const std::vector<std::vector<float>> *p)
	{
		m_data.ptr = (void *)p;
		m_type = rds_type::flt_arr_arr;
		return *this;
	}

private:
	void write_namespace(std::ofstream &out)
	{
		static const char *ns = "names";
		write_int<uint16_t>(out, 0);
		write_int<uint16_t>(out, 0x402);
		if(nullptr == ns)
		{
			write_int<uint32_t>(out, 0x1FF);
		}
		else
		{
			write_int<uint32_t>(out, 1);
			write_str(out, ns);
			ns = nullptr;
		}
	}

	void write_str(std::ofstream &out, const char *s)
	{
		write_int<uint16_t>(out, 0x04);
		write_int<uint16_t>(out, 0x09);
		write_str_plain(out, s);
	}

	void write_str_plain(std::ofstream &out, const char *s)
	{
		write_int<uint32_t>(out, strlen(s));
		out.write(s, strlen(s));
	}

	void write_float_array(std::ofstream &out, const std::vector<float> &f)
	{
		write_int<uint16_t>(out, 0x00);
		write_int<uint16_t>(out, 0x0E);
		write_int<uint32_t>(out, f.size());
		for(auto &ref : f)
		{
			double d = ref;
			uint64_t x = *((uint64_t *)&d);
			write_int<uint64_t>(out, x);
		}
	}

	void save_inner(std::ofstream &out)
	{
		if(rds_type::object == m_type)
		{
			write_int<uint16_t>(out, 0x00);
			write_int<uint16_t>(out, 0x213);
			write_int<uint32_t>(out, m_children.size());
			for(auto &ref : m_children)
			{
				ref.second->save_inner(out);
			}
			write_namespace(out);
			// String array
			write_int<uint16_t>(out, 0x00);
			write_int<uint16_t>(out, 0x10);
			write_int<uint32_t>(out, m_children.size());
			for(auto &ref: m_children)
			{
				write_str(out, ref.first.c_str());
			}
			write_int<uint32_t>(out, 0xFE);
		}
		else if(rds_type::flt == m_type)
		{
			write_int<uint16_t>(out, 0x00);
			write_int<uint16_t>(out, 0x0E);
			write_int<uint32_t>(out, 1);
			write_int<uint64_t>(out, m_data.v);
		}
		else if(rds_type::flt_arr == m_type)
		{
			auto &v = *(const std::vector<float> *)m_data.ptr;
			write_float_array(out, v);
		}
		else if(rds_type::flt_arr_arr == m_type)
		{
			auto &v = *(const std::vector<std::vector<float>> *)m_data.ptr;
			write_int<uint16_t>(out, 0x00);
			write_int<uint16_t>(out, 0x13);
			write_int<uint32_t>(out, v.size());
			for(auto &ref : v)
			{
				write_float_array(out, ref);
			}
		}
		else
		{
			throw std::runtime_error("Cant serialize type");
		}
	}

public:
	void save(std::ofstream &out)
	{
		write_int<uint32_t>(out, 0x580a0000);
		write_int<uint16_t>(out, 0x0003); // 3
		write_int<uint16_t>(out, 0x0003); // 3
		write_int<uint16_t>(out, 0x0602); // 206
		write_int<uint16_t>(out, 0x0003); // 3
		write_int<uint16_t>(out, 0x0500); // 5

		write_str_plain(out, "UTF-8");

		save_inner(out);
	}

	rds_type m_type;
	union
	{
		void *ptr; //make uniion ;9
		uint64_t v;
	}
	m_data;
	std::unordered_map<std::string, std::shared_ptr<rds_data>> m_children;
};


static bool read_object(std::ifstream &in, std::string indent)
{
	uint16_t u0 = read_int<uint16_t>(in);
	uint16_t type = read_int<uint16_t>(in);

	std::cout << indent << std::hex << "U0: " << u0 << ", TYPE: " << type << std::endl;

	if(0x213 == type)
	{
		// Object?
		uint32_t num_items = read_int<uint32_t>(in);

		std::cout << indent << "<Object[" << num_items << "]>" << std::endl;

		for(uint32_t i = 0; i < num_items; ++i)
		{
			read_object(in, indent + "  ");
		}
		if(7 == num_items)
		{
			std::cout << "DONEZON" << std::endl;
		}
		read_object(in, indent + "  ");
		read_object(in, indent + "  ");
		if(!read_object(in, indent + "  "))
		{
			throw std::runtime_error("Failure");
		}
	}
	else if(0x13 == type)
	{
		// Array? Of arrays?
		uint32_t num_items = read_int<uint32_t>(in);
		std::cout << indent << "Array[" << num_items << "]" << std::endl;
		for(uint32_t i = 0; i < num_items; ++i)
		{
			read_object(in, indent + "  ");
		}
	}
	else if(0x0E == type)
	{
		uint32_t num_items = read_int<uint32_t>(in);
		std::cout << indent << "Flt(" << num_items << "): ";
		for(uint32_t i = 0; i < num_items; ++i)
		{
			std::cout << read_double(in) << ", ";
		}
		std::cout << std::endl;
	}
	else if(0x402 == type)
	{
		uint32_t num_items = read_int<uint32_t>(in);
		std::cout << indent << "<NAME_THING[" << num_items << "]>" << std::endl;
		if(0x1FF != num_items)
		{
			read_object(in, indent + "  ");
		}
	}
	else if(0x09 == type)
	{
		if(0x04 != u0)
		{
			throw std::runtime_error("Strange string?");
		}
		std::cout << indent << "STR: '" << read_str(in) << "'" << std::endl;
	}
	else if(0x10 == type)
	{
		uint32_t num_items = read_int<uint32_t>(in);
		std::cout << indent << "<StringArray[" << num_items << "]>" << std::endl;
		if(0x00 != u0)
		{
			throw std::runtime_error("Strange key thing?");
		}
		for(uint32_t i = 0; i < num_items; ++i)
		{
			read_object(in, indent + "  ");
		}
	}
	else if(0xFE == type)
	{
		if(0x00 != u0)
		{
			throw std::runtime_error("Strange end thing");
		}
		std::cout << indent << "<EndOfObject>" << std::endl;
		return true;
	}
	else
	{
		throw std::runtime_error("fiskapa");
	}
	return false;
}

static void read_names(std::ifstream &in)
{
	//
	// Hack fuck
	//
	if(0x402 != read_int<uint32_t>(in))
	{
		throw std::runtime_error("!402");
	}

	auto id = read_int<uint32_t>(in);
	std::cout << "ID-ish: " << id << std::endl;
	if(1 == id)
	{
		auto type = read_int<uint32_t>(in);
		if(type != 0x40009)
		{
			throw std::runtime_error("Name not string");
		}
		std::cout << read_str(in) << std::endl;
	}
	auto u0 = read_int<uint32_t>(in);
	auto count = read_int<uint32_t>(in);
	std::cout << "U0: " << u0 << std::endl;
	std::cout << "Number of names in node: " << count << std::endl;
	for(uint32_t i = 0; i < count; ++i)
	{
		auto type = read_int<uint32_t>(in);
		if(0x40009 != type)
		{
			throw std::runtime_error("Name not string 2");
		}
		std::cout << "Name[" << i << "]: " << read_str(in) << std::endl;
	}
	if(0x000000FE != read_int<uint32_t>(in))
	{
		throw std::runtime_error("Missing end marker");
	}
}

	/*

00 00 04 02 00 00 00 01
	00 04 00 09 00 00 00 05 6e 61 6d 65 73 \ ; "names"
	00 00 00 10 00 00 00 01 \
	00 04 00 09 00 00 00 07 6e 65 73 74 73 6b 6f \ ; "nestedsko"
00 00 00 fe

00 00 04 02 00 00 01 ff \
	00 00 00 10 00 00 00 04 \
	00 04 00 09 00 00 00 03 73 6b 6f \
	00 04 00 09 00 00 00 05 66 6c 6f 61 74 \
	00 04 00 09 00 00 00 05 6c 69 73 74 61 \
	00 04 00 09 00 00 00 06 6e 65 73 74 65 64 \
00 00 00 fe
*/

int test(int argc, const char *argv[])
{
	try
	{
		if(2 != argc)
		{
			throw std::runtime_error("What file?");
		}
		std::ifstream in{argv[1], std::ios_base::binary};
		if(!in.is_open())
		{
			throw std::runtime_error("Failed to open file.");
		}

		if(0x580a0000 != read_int<uint32_t>(in))
		{
			throw std::runtime_error("Invalid magic");
		}

		read_int<uint16_t>(in); // 3
		read_int<uint16_t>(in); // 3
		read_int<uint16_t>(in); // 206
		read_int<uint16_t>(in); // 3
		read_int<uint16_t>(in); // 5

		auto format{read_str(in)};

		std::cout << "Format: " << format << std::endl;
		read_object(in, "");

	}
	catch(const std::exception &e)
	{
		std::cout << "Failure :(" << std::endl;
		std::cout << e.what() << std::endl;
	}
	rds_data d;
	
	std::vector<std::vector<float>> actions
	{
		{
			{ 0, 0, 1, 1, 0, 1, 1, 1 },
			{ 0, 0, 0, 1, 0, 0, 1, 0 },
			{ 0, 0, 0, 1, 0, 1, 1, 0 }
		}
	};

	std::vector<float> bn1_bias{0.00874151661992073, -0.0007319115102291107};
	d["actions"] = &actions;
	d["dparameters"]["bn1.bias"]["values"] = &bn1_bias;
	d["kalle"]["hampe"] = 5.9;

	std::ofstream out("test.raw", std::ios_base::binary);
	d.save(out);
	// system("gzip test.raw");
	return 0;
}



