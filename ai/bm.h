#pragma once

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <string>
struct CumBenchmark {
	struct Report {
		Report()
		: time_total{0}
		, count{0} {}
		std::chrono::duration<double> time_total;
		size_t count;
	};

	void report(const std::string &name, const std::chrono::duration<double> dur) {
		auto &b = benchmarks[name];
		b.time_total += dur;
		++b.count;
	}

	void print() {
		for(auto& b: benchmarks) {
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(b.second.time_total).count();
			std::cout << "[CBench] " << b.first << ": " << static_cast<float>(ms) / b.second.count 
					  << " (" << ms << " over " << b.second.count << ")" << std::endl;
		}
	}
	std::unordered_map<std::string, Report> benchmarks;
};

struct Benchmark {
	Benchmark(CumBenchmark &cb, std::string &&n)
		: start{std::chrono::high_resolution_clock::now()}
		, name(std::move(n))
		, cum(&cb) 
		, stopped(false)
	{}
	Benchmark(std::string &&n)
		: start{std::chrono::high_resolution_clock::now()}
		, name{std::move(n)}
		, cum(nullptr)
		, stopped(false)
	 {
		//  std::cout << "[Bench] " << name << std::endl;
	 }
	~Benchmark() {
		stop();
	}

	void stop() {
		if(stopped) {
			return;
		}
		stopped = true;
		auto dur = std::chrono::high_resolution_clock::now() - start;
		if(!cum) {
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
			std::cout << "[Bench] " << name << " took " << ms << " ms" << std::endl;
		}
		else {
			cum->report(name, dur);
		}
	}
	std::chrono::time_point<std::chrono::system_clock> start;
	std::string name;
	CumBenchmark *cum;
	bool stopped;
};