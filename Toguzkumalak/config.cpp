#include "pch.h"
#include "config.h"
#include <fstream>

Config::Config(const std::string& path) {
    load(path);
}

void Config::set(const std::string& key, std::vector<std::string> value, const std::string& section){
    sections_[section][key] = std::move(value);
}

void Config::showAll(){
    for (const auto& sec : sections_) {
        printf("[%s]\n", sec.first.c_str());
        for (const auto& val : sec.second) {
            printf("%s ", val.first.c_str());
            for (const auto& param : val.second) {
                printf("%s ", param.c_str());
            }
            printf("\n");
        }
    }
}

void Config::load(const std::string& path) {
    sections_.clear();
    active_section_ = "global";
    std::ifstream file(path);
    if (!file.is_open()) {
        printf("Error opening config file\n");
    }
    std::string line;
    std::string section = "global";
    while (std::getline(file, line)) {
        // trim
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }
        std::string key;
        std::string param;
        std::vector<std::string> params;
        std::istringstream iss(line);
        iss >> key;
        while (iss >> param) {
            if (param[0] == '#' || param[0] == ';') break;
            params.push_back(param);
        }
        set(key, params, section);
    }
}

void Config::setSection(const std::string& section) {
    active_section_ = section;
}

