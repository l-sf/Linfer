
#include "ilogger.hpp"
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <cmath>
#include <string>
#include <mutex>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>
#include <stack>
#include <functional>
#include <signal.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>


#define __GetTimeBlock						\
    time_t timep;							\
    time(&timep);							\
    tm& t = *(tm*)localtime(&timep);


namespace iLogger{

    using namespace std;

    const char* level_string(LogLevel level){
        switch (level){
            case LogLevel::Debug: return "debug";
            case LogLevel::Verbose: return "verbo";
            case LogLevel::Info: return "info";
            case LogLevel::Warning: return "warn";
            case LogLevel::Error: return "error";
            case LogLevel::Fatal: return "fatal";
            default: return "unknow";
        }
    }

    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f*s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i) {
        case 0:r = v; g = t; b = p;break;
        case 1:r = q; g = v; b = p;break;
        case 2:r = p; g = v; b = t;break;
        case 3:r = p; g = q; b = v;break;
        case 4:r = t; g = p; b = v;break;
        case 5:r = v; g = p; b = q;break;
        default:r = 1; g = 1; b = 1;break;}
        return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
        float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
        float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
        return hsv2bgr(h_plane, s_plane, 1);
    }

    string date_now() {
        char time_string[20];
        __GetTimeBlock;

        sprintf(time_string, "%04d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
        return time_string;
    }

    string time_now(){
        char time_string[20];
        __GetTimeBlock;

        sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
        return time_string;
    }

    size_t file_size(const string& file){
        struct stat st{};
        stat(file.c_str(), &st);
        return st.st_size;
    }

    time_t last_modify(const string& file){
        struct stat st{};
        stat(file.c_str(), &st);
        return st.st_mtim.tv_sec;
    }

    int get_month_by_name(char* month){
        if(strcmp(month,"Jan") == 0)return 0;
        if(strcmp(month,"Feb") == 0)return 1;
        if(strcmp(month,"Mar") == 0)return 2;
        if(strcmp(month,"Apr") == 0)return 3;
        if(strcmp(month,"May") == 0)return 4;
        if(strcmp(month,"Jun") == 0)return 5;
        if(strcmp(month,"Jul") == 0)return 6;
        if(strcmp(month,"Aug") == 0)return 7;
        if(strcmp(month,"Sep") == 0)return 8;
        if(strcmp(month,"Oct") == 0)return 9;
        if(strcmp(month,"Nov") == 0)return 10;
        if(strcmp(month,"Dec") == 0)return 11;
        return -1;
    }

    int get_week_day_by_name(char* wday){
        if(strcmp(wday,"Sun") == 0)return 0;
        if(strcmp(wday,"Mon") == 0)return 1;
        if(strcmp(wday,"Tue") == 0)return 2;
        if(strcmp(wday,"Wed") == 0)return 3;
        if(strcmp(wday,"Thu") == 0)return 4;
        if(strcmp(wday,"Fri") == 0)return 5;
        if(strcmp(wday,"Sat") == 0)return 6;
        return -1;
    }

    string gmtime(time_t t){
        t += 28800;
        tm* gmt = ::gmtime(&t);

        // http://en.cppreference.com/w/c/chrono/strftime
        // e.g.: Sat, 22 Aug 2015 11:48:50 GMT
        //       5+   3+4+   5+   9+       3   = 29
        const char* fmt = "%a, %d %b %Y %H:%M:%S GMT";
        char tstr[30];
        strftime(tstr, sizeof(tstr), fmt, gmt);
        return tstr;
    }

    string gmtime_now() {
        return gmtime(time(nullptr));
    }

    bool mkdir(const string& path){
        return ::mkdir(path.c_str(), 0755) == 0;
    }

    bool mkdirs(const string& path){

        if (path.empty()) return false;
        if (exists(path)) return true;

        string _path = path;
        char* dir_ptr = (char*)_path.c_str();
        char* iter_ptr = dir_ptr;
        
        bool keep_going = *iter_ptr not_eq 0;
        while (keep_going){

            if (*iter_ptr == 0)
                keep_going = false;

            if ((*iter_ptr == '/' and iter_ptr not_eq dir_ptr) or *iter_ptr == 0){
                char old = *iter_ptr;
                *iter_ptr = 0;
                if (!exists(dir_ptr)){
                    if (!mkdir(dir_ptr)){
                        if(!exists(dir_ptr)){
                            INFOE("mkdirs %s return false.", dir_ptr);
                            return false;
                        }
                    }
                }
                *iter_ptr = old;
            }
            iter_ptr++;
        }
        return true;
    }

    bool isfile(const string& file){

        struct stat st;
        stat(file.c_str(), &st);
        return S_ISREG(st.st_mode);

    }

    FILE* fopen_mkdirs(const string& path, const string& mode){

        FILE* f = fopen(path.c_str(), mode.c_str());
        if (f) return f;

        int p = path.rfind('/');

        if (p == -1)
            return nullptr;
        
        string directory = path.substr(0, p);
        if (!mkdirs(directory))
            return nullptr;

        return fopen(path.c_str(), mode.c_str());
    }

    bool exists(const string& path){
        return access(path.c_str(), R_OK) == 0;
    }

    string format(const char* fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

    string file_name(const string& path, bool include_suffix){

        if (path.empty()) return "";

        int p = path.rfind('/');

        p += 1;

        //include suffix
        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    long long timestamp_now() {
        return chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    }

    static struct Logger{
        mutex logger_lock_;
        string logger_directory;
        LogLevel logger_level{LogLevel::Info};
        vector<string> cache_, local_;
        shared_ptr<thread> flush_thread_;
        atomic<bool> keep_run_{false};
        shared_ptr<FILE> handler;
        bool logger_shutdown{false};

        void write(const string& line) {

            lock_guard<mutex> l(logger_lock_);
            if(logger_shutdown) 
                return;

            if (!keep_run_) {

                if(flush_thread_) 
                    return;

                cache_.reserve(1000);
                keep_run_ = true;
                flush_thread_.reset(new thread(std::bind(&Logger::flush_job, this)));
            }
            cache_.emplace_back(line);
        }

        void flush() {

            if (cache_.empty())
                return;

            {
                std::lock_guard<mutex> l(logger_lock_);
                std::swap(local_, cache_);
            }

            if (!local_.empty() and !logger_directory.empty()) {

                auto now = date_now();
                auto file = format("%s%s.txt", logger_directory.c_str(), now.c_str());
                if (!exists(file)) {
                    handler.reset(fopen_mkdirs(file, "wb"), fclose);
                }
                else if (!handler) {
                    handler.reset(fopen_mkdirs(file, "a+"), fclose);
                }

                if (handler) {
                    for (auto& line : local_)
                        fprintf(handler.get(), "%s\n", line.c_str());
                    fflush(handler.get());
                    handler.reset();
                }
            }
            local_.clear();
        }

        void flush_job() {

            auto tick_begin = timestamp_now();
            std::vector<string> local;
            while (keep_run_) {

                if (timestamp_now() - tick_begin < 1000) {
                    this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                tick_begin = timestamp_now();
                flush();
            }
            flush();
        }

        void set_save_directory(const string& loggerDirectory) {
            logger_directory = loggerDirectory;

            if (logger_directory.empty())
                logger_directory = ".";

            if (logger_directory.back() not_eq '/') {
                logger_directory.push_back('/');
            }

        }

        void set_logger_level(LogLevel level){
            logger_level = level;
        }

        void close(){
            {
                lock_guard<mutex> l(logger_lock_);
                if (logger_shutdown) return;

                logger_shutdown = true;
            };

            if (!keep_run_) return;
            keep_run_ = false;
            flush_thread_->join();
            flush_thread_.reset();
            handler.reset();
        }

        virtual ~Logger(){
            close();
        }
    }__g_logger;

    void destroy_logger(){
        __g_logger.close();
    }

    static void remove_color_text(char* buffer){
        
        //"\033[31m%s\033[0m"
        char* p = buffer;
        while(*p){

            if(*p == 0x1B){
                char np = *(p + 1);
                if(np == '['){
                    // has token
                    char* t = p + 2;
                    while(*t){
                        if(*t == 'm'){
                            t = t + 1;
                            char* k = p;
                            while(*t){
                                *k++ = *t++;
                            }
                            *k = 0;
                            break;
                        }
                        t++;
                    }
                }
            }
            p++;
        }
    }


    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...) {

        if(level > __g_logger.logger_level)
            return;

        string now = time_now();
        va_list vl;
        va_start(vl, fmt);
        
        char buffer[2048];
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Warning) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Info) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Verbose) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]", level_string(level));
        }
        else {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
        }

        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            fprintf(stderr, "%s\n", buffer);
        }
        else if (level == LogLevel::Warning) {
            fprintf(stdout, "%s\n", buffer);
        }
        else {
            fprintf(stdout, "%s\n", buffer);
        }

        if(!__g_logger.logger_directory.empty()){
            // remove save color txt
            remove_color_text(buffer);
            __g_logger.write(buffer);
            if (level == LogLevel::Fatal) {
                __g_logger.flush();
            }
        }

        if (level == LogLevel::Fatal) {
            fflush(stdout);
            abort();
        }
    }


    bool alphabet_equal(char a, char b, bool ignore_case){
        if (ignore_case){
            a = a > 'a' and a < 'z' ? a - 'a' + 'A' : a;
            b = b > 'a' and b < 'z' ? b - 'a' + 'A' : b;
        }
        return a == b;
    }

    static bool pattern_match_body(const char* str, const char* matcher, bool igrnoe_case){
        //   abcdefg.pnga          *.png      > false
        //   abcdefg.png           *.png      > true
        //   abcdefg.png          a?cdefg.png > true

        if (!matcher or !*matcher or !str or !*str) return false;

        const char* ptr_matcher = matcher;
        while (*str){
            if (*ptr_matcher == '?'){
                ptr_matcher++;
            }
            else if (*ptr_matcher == '*'){
                if (*(ptr_matcher + 1)){
                    if (pattern_match_body(str, ptr_matcher + 1, igrnoe_case))
                        return true;
                }
                else{
                    return true;
                }
            }
            else if (!alphabet_equal(*ptr_matcher, *str, igrnoe_case)){
                return false;
            }
            else{
                if (*ptr_matcher)
                    ptr_matcher++;
                else
                    return false;
            }
            str++;
        }

        while (*ptr_matcher){
            if (*ptr_matcher not_eq '*')
                return false;
            ptr_matcher++;
        }
        return true;
    }

    bool pattern_match(const char* str, const char* matcher, bool igrnoe_case){
        //   abcdefg.pnga          *.png      > false
        //   abcdefg.png           *.png      > true
        //   abcdefg.png          a?cdefg.png > true

        if (!matcher or !*matcher or !str or !*str) return false;

        char filter[500];
        strcpy(filter, matcher);

        vector<const char*> arr;
        char* ptr_str = filter;
        char* ptr_prev_str = ptr_str;
        while (*ptr_str){
            if (*ptr_str == ';'){
                *ptr_str = 0;
                arr.push_back(ptr_prev_str);
                ptr_prev_str = ptr_str + 1;
            }
            ptr_str++;
        }

        if (*ptr_prev_str)
            arr.push_back(ptr_prev_str);

        for (int i = 0; i < arr.size(); ++i){
            if (pattern_match_body(str, arr[i], igrnoe_case))
                return true;
        }
        return false;
    }

    vector<string> find_files(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory)
    {
        string realpath = directory;
        if (realpath.empty())
            realpath = "./";

        char backchar = realpath.back();
        if (backchar not_eq '\\' and backchar not_eq '/')
            realpath += "/";

        struct dirent* fileinfo;
        DIR* handle;
        stack<string> ps;
        vector<string> out;
        ps.push(realpath);

        while (!ps.empty())
        {
            string search_path = ps.top();
            ps.pop();

            handle = opendir(search_path.c_str());
            if (handle not_eq 0)
            {
                while (fileinfo = readdir(handle))
                {
                    struct stat file_stat;
                    if (strcmp(fileinfo->d_name, ".") == 0 or strcmp(fileinfo->d_name, "..") == 0)
                        continue;

                    if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
                        continue;

                    if (!findDirectory and !S_ISDIR(file_stat.st_mode) or
                        findDirectory and S_ISDIR(file_stat.st_mode))
                    {
                        if (pattern_match(fileinfo->d_name, filter.c_str()))
                            out.push_back(search_path + fileinfo->d_name);
                    }

                    if (includeSubDirectory and S_ISDIR(file_stat.st_mode))
                        ps.push(search_path + fileinfo->d_name + "/");
                }
                closedir(handle);
            }
        }
        return out;
    }


    string align_blank(const string& input, int align_size, char blank){
        if(input.size() >= align_size) return input;
        string output = input;
        for(int i = 0; i < align_size - input.size(); ++i)
            output.push_back(blank);
        return output;
    }


    bool save_file(const string& file, const void* data, size_t length, bool mk_dirs){

        if (mk_dirs){
            int p = (int)file.rfind('/');

            if (p not_eq -1){
                if (!mkdirs(file.substr(0, p)))
                    return false;
            }
        }

        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data and length > 0){
            if (fwrite(data, 1, length, f) not_eq length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }

    bool save_file(const string& file, const string& data, bool mk_dirs){
        return save_file(file, data.data(), data.size(), mk_dirs);
    }

    bool save_file(const string& file, const vector<uint8_t>& data, bool mk_dirs){
        return save_file(file, data.data(), data.size(), mk_dirs);
    }


}; // namespace Logger