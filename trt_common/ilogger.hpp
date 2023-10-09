
#ifndef ILOGGER_HPP
#define ILOGGER_HPP

#include <string>
#include <vector>
#include <tuple>
#include <ctime>


namespace iLogger{

    using namespace std;

    enum class LogLevel : int{
        Debug   = 5,
        Verbose = 4,
        Info    = 3,
        Warning = 2,
        Error   = 1,
        Fatal   = 0
    };

    #define INFOD(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Debug, __VA_ARGS__)
    #define INFOV(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Verbose, __VA_ARGS__)
    #define INFO(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Info, __VA_ARGS__)
    #define INFOW(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Warning, __VA_ARGS__)
    #define INFOE(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Error, __VA_ARGS__)
    #define INFOF(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Fatal, __VA_ARGS__)

    string date_now();
    string time_now();
    string gmtime_now();
	string gmtime(time_t t);

    bool isfile(const string& file);
    bool mkdir(const string& path);
    bool mkdirs(const string& path);
    bool exists(const string& path);
    string format(const char* fmt, ...);
    FILE* fopen_mkdirs(const string& path, const string& mode);
    string file_name(const string& path, bool include_suffix=true);
    long long timestamp_now();

    // return, 0-255, 0-255, 0-255
    tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

    //   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true
	bool pattern_match(const char* str, const char* matcher, bool igrnoe_case = true);
    vector<string> find_files(
        const string& directory, 
        const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

    string align_blank(const string& input, int align_size, char blank=' ');
    bool save_file(const string& file, const vector<uint8_t>& data, bool mk_dirs = true);
    bool save_file(const string& file, const string& data, bool mk_dirs = true);
	bool save_file(const string& file, const void* data, size_t length, bool mk_dirs = true);


    // 关于logger的api
    const char* level_string(LogLevel level);
    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);
    void destroy_logger();

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align;}
}


#endif // ILOGGER_HPP