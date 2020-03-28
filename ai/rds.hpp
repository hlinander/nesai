#pragma once

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

#if 0
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
#endif

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
		flt_vec,
		flt_vec_vec,
		flt_vec_arr16
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
		m_type = rds_type::flt_vec;
		return *this;
	}

	rds_data &operator =(const std::vector<std::vector<float>> *p)
	{
		m_data.ptr = (void *)p;
		m_type = rds_type::flt_vec_vec;
		return *this;
	}

	rds_data &operator =(const std::vector<std::array<uint8_t, 16>> *p)
	{
		m_data.ptr = (void *)p;
		m_type = rds_type::flt_vec_arr16;
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

	template <class T>
	void write_float_array(std::ofstream &out, const T &f)
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
		else if(rds_type::flt_vec == m_type)
		{
			auto &v = *(const std::vector<float> *)m_data.ptr;
			write_float_array(out, v);
		}
		else if(rds_type::flt_vec_vec == m_type)
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
		else if(rds_type::flt_vec_arr16 == m_type)
		{
			auto &v = *(const std::vector<std::array<uint8_t, 16>> *)m_data.ptr;
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


