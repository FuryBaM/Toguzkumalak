#pragma once
#include <mutex>

class Config {
public:
	explicit Config(const std::string& path) {
		load(path);
	}
	void load(const std::string& path);
	template<typename T>
	std::vector<T> get(const std::string& key, const std::vector<T>& default_value) const;
	template<typename T>
	T get(const std::string& key, const T& default_value, size_t index) const;
	void showAll();
private:
	std::unordered_map<std::string, std::vector<std::string>> data_;
	Config(const Config&) = delete;
	Config& operator=(const Config&) = delete;
	void set(const std::string& key, std::vector<std::string> value);
};

template<typename T>
inline T process(const std::string& value) {
	std::cout << "Not allowed" << value << std::endl;
	if constexpr (std::is_pointer<T>::value) {
		return nullptr;
	}
	else {
		return T();
	}
}

template<>
inline int process<int>(const std::string& value) {
	try {
		int result = std::stoi(value);
		return result;
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
	catch (const std::out_of_range& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
}

template<>
inline float process<float>(const std::string& value) {
	try {
		float result = std::stof(value);
		return result;
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
	catch (const std::out_of_range& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
}

template<>
inline double process<double>(const std::string& value) {
	try {
		double result = std::stod(value);
		return result;
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
	catch (const std::out_of_range& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
}

template<>
inline std::string process<std::string>(const std::string& value) {
	return value;
}

template<>
inline long process<long>(const std::string& value) {
	try {
		long result = std::stol(value);
		return result;
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
	catch (const std::out_of_range& e) {
		std::cerr << e.what() << "\n";
		std::exit(1);
	}
}

template<>
inline bool process<bool>(const std::string& value) {
	std::string lower_value = value;
	std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);

	if (lower_value == "1" || lower_value == "true") {
		return true;
	}
	else if (lower_value == "0" || lower_value == "false") {
		return false;
	}
	else {
		std::cout << "Incorrect bool value" << value << std::endl;
		return false;
	}
}

template<typename T>
inline std::vector<T> Config::get(const std::string& key, const std::vector<T>& default_value) const {
	std::vector<T> values;
	if (data_.find(key) != data_.end()) {
		std::vector<std::string> params = data_.at(key);
		for (const auto& value : params) {
			values.push_back(process<T>(value));
		}
	}
	else {
		values = default_value;
	}
	return values;
}

template<typename T>
inline T Config::get(const std::string& key, const T& default_value, size_t index) const {
	if (data_.find(key) != data_.end()) {
		std::vector<std::string> params = data_.at(key);
		if (index < params.size()) {
			return process<T>(params[index]);
		}
		else {
			return default_value;
		}
	}
	else {
		return default_value;
	}
}