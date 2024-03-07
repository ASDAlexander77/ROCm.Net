// api.h - Contains declarations of api functions
#pragma once

#ifndef API_FILE_123
#define API_FILE_123

#ifdef HIPLIBRARY_EXPORTS
#define HIPLIBRARY_API __declspec(dllexport)
#else
#define HIPLIBRARY_API __declspec(dllimport)
#endif

extern "C" HIPLIBRARY_API int compile(const char*, const char*, const char**, int, const char**, int);

extern "C" HIPLIBRARY_API size_t get_log(const char*, char*, size_t);

extern "C" HIPLIBRARY_API size_t get_size(const char*);

extern "C" HIPLIBRARY_API int param_int(const char*, const char*, int);

extern "C" HIPLIBRARY_API int param_float(const char*, const char*, float);

extern "C" HIPLIBRARY_API int param_float_array(const char*, const char*, const float*, size_t);

extern "C" HIPLIBRARY_API int run(const char*, const char*);

extern "C" HIPLIBRARY_API int result_float_array(const char*, const char*, float**, size_t);

extern "C" HIPLIBRARY_API int clear_run(const char*);

#endif // API_FILE_123