#include "pch.h"
#include "config.h"

void Config::set(const std::string& key, std::vector<std::string> value){
	data_[key] = value;
}

void Config::showAll(){
	for (const auto& val : data_) {
		printf("%s ", val.first.c_str());
		for (const auto& param : val.second) {
			printf("%s ", param.c_str());
		}
		printf("\n");
	}
}

void Config::load(const std::string& path) {
	data_.clear();
	std::ifstream file(path);
	if (!file.is_open()) {
		printf("Error opening config file\n");
	}
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::string key;
		std::string param;
		std::vector<std::string> params;
		std::istringstream iss(line);
		iss >> key;
		while (iss >> param) {
			if (param[0] == '#') break;
			params.push_back(param);
		}
		set(key, params);
	}
}
