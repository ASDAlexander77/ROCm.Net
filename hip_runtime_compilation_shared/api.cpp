#include "api.h"

#include "utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <string>
#include <vector>
#include <map>

struct HipCodeParam
{
    HipCodeParam() : size{0}, aligned_size{0}, offset{0}, allocated{nullptr}, is_array{false} {};

    HipCodeParam(size_t size_, size_t aligned_size_, size_t offset_, void* allocated_, bool is_array_) : 
        size{size_}, aligned_size{aligned_size_}, offset{offset_}, allocated{allocated_}, is_array{is_array_} {};

public:
    size_t size;
    size_t aligned_size;
    size_t offset;
    void* allocated;
    bool is_array;
};

struct HipCodeArgs
{
public:
    HipCodeArgs() : params{}, size{0}, args{} {};

    template <typename T>
    int param(std::string param_name, T scalarValue, bool is_array = false, void* allocated = nullptr)
    {
        auto it = params.find(param_name);
        if (it != std::end(params))
        {
            auto &param = (*it).second;
            if (param.is_array)
            {
                if (param.allocated != nullptr)
                    HIP_CHECK(hipFree(param.allocated));
                param.allocated = allocated;
            }

            // already added
            set<T>(param.offset, scalarValue);
        }
        else
        {
            append<T>(param_name, scalarValue, is_array, allocated);
        }

        return HIPRTC_SUCCESS;
    }

    template <typename T>
    int param_array(const char* param_name, const T* array_value, size_t count)
    {
        // size
        if (count > size)
        {
            size = count;
        }

        // array pointer
        size_t size_bytes = count * sizeof(T);
        T* d_arrayValue{};

        HIP_CHECK(hipMalloc(&d_arrayValue, size_bytes));
        HIP_CHECK(hipMemcpy(d_arrayValue, array_value, size_bytes, hipMemcpyHostToDevice));

        return param(param_name, d_arrayValue, true, d_arrayValue);
    }

    template <typename T>
    int result_array(std::string param_name, T* return_array, size_t count)
    {
        // size for Float array
        size_t size_bytes = size * sizeof(T);

        auto it = params.find(param_name);
        if (it == std::end(params))
        {
            return -1;
        }

        auto offset = (*it).second.offset;

        auto d_pointer = get<T*>(offset);

        // Copy results from device to host.
        HIP_CHECK(hipMemcpy(return_array, d_pointer, size_bytes, hipMemcpyDeviceToHost));

        return HIPRTC_SUCCESS;
    }

    size_t get_args_size()
    {
        return args.size();
    }

    char* get_args_data()
    {
        return args.data();
    }

    int free()
    {
        // Free device memory.
        for (auto& ptr : params)
        {
            if (ptr.second.is_array)
                HIP_CHECK(hipFree(ptr.second.allocated));
        }

        return HIPRTC_SUCCESS;
    }

    size_t size;

private:

    template <typename T>
    void set(size_t offset, T value)
    {
        *(reinterpret_cast<T*>(&(args.data()[offset]))) = value;
    }

    template <typename T>
    T get(size_t offset)
    {
        return *(reinterpret_cast<T*>(&(args.data()[offset])));
    }

    template <typename T>
    size_t append(std::string name, T value, bool is_array, void* allocated)
    {
        auto offset = args.size();
        auto delta = std::max(sizeof(value), sizeof(void*)); // aligning fix for CUDA executions
        auto newSize = offset + delta;

        if (newSize > args.size())
        {
            args.resize(newSize);
        }

        set(offset, value);

        params[name] = HipCodeParam(sizeof(value), delta, offset, allocated, is_array);

        return offset;
    }

    std::map<std::string, HipCodeParam> params;
    std::vector<char> args;
};

class HipCode
{
public:
    HipCode(const char* kernel_name_) : kernel_name(kernel_name_), code{}, log{}, args{}, module(nullptr) {};

    HipCodeArgs& get_args()
    {
        return args;
    }

    int compile(const char* kernel_code, const char** header_names, int header_names_count, const char** header_sources, int header_sources_count)
    {
        // Program to be compiled in runtime.
        hiprtcProgram prog;

        // Create program.
        hiprtcCreateProgram(&prog,
                            kernel_code,
                            kernel_name.data(),
                            header_sources_count,
                            header_sources,
                            header_names);

        // Get device properties from the first device available.
        hipDeviceProp_t        props;
        constexpr unsigned int device_id = 0;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));

        std::vector<const char*> options;

        // Obtain architecture's name from device properties and initialize array of compile options. When in CUDA we omit this option.
    #ifdef __HIP_PLATFORM_AMD__
        std::string arch_option;
        if(props.gcnArchName[0])
        {
            arch_option = std::string("--gpu-architecture=") + props.gcnArchName;
            options.push_back(arch_option.c_str());
        }
    #endif

        // Compile program in runtime. Parameters are the program, number of options and array with options.
        const hiprtcResult compile_result{hiprtcCompileProgram(prog, options.size(), options.data())};

        // Get the size of the log (possibly) generated during the compilation.
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);

        // If the compilation generated a log, print it.
        if(log_size)
        {
            log.resize(log_size);
            hiprtcGetProgramLog(prog, &log[0]);
        }

        // If the compilation failed, say so and exit.
        if(compile_result != HIPRTC_SUCCESS)
        {
            return compile_result;
        }

        // Get the size (in number of characters) of the binary compiled from the program.
        size_t code_size;
        hiprtcGetCodeSize(prog, &code_size);

        // Store compiled binary as a vector of characters.
        code.resize(code_size);
        hiprtcGetCode(prog, code.data());

        // Destroy program object.
        hiprtcDestroyProgram(&prog);

        return HIPRTC_SUCCESS;
    }

    size_t get_log(char* log_out, size_t allocated_size)
    {
        if (log.size() <= allocated_size)
        {
            std::copy(log.begin(), log.end(), log_out);
        }

        return log.size();
    }

    size_t get_size()
    {
        return code.size();
    }

    int run(const char* function_name)
    {
        auto offset = args.get_args_size();
        // Total number of float elements in each device vector.
        auto size = args.size;
        auto args_data = args.get_args_data();

        // Now we launch the kernel on the device.

        // Number of threads per kernel block.
        constexpr unsigned int block_size = 128;

        // Number of blocks per kernel grid, calculated as ceil(size/block_size).
        unsigned int grid_size = (size + block_size - 1) / block_size;

        // Load the HIP module corresponding to the compiled binary into the current context.
        if (!module)
        {
            HIP_CHECK(hipModuleLoadData(&module, code.data()));
        }

        // Extract kernel from module into a function object.
        hipFunction_t kernel;
        HIP_CHECK(hipModuleGetFunction(&kernel, module, function_name));

        // Create array with kernel arguments and its size.
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          args_data,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &offset,
                          HIP_LAUNCH_PARAM_END};

        //std::cout << "Calculating y[i] = a * x[i] + y[i] over " << size << " elements." << std::endl;

        // Launch the kernel on the NULL stream and with the above configuration.
        HIP_CHECK(hipModuleLaunchKernel(kernel,
                                        grid_size,
                                        1,
                                        1,
                                        block_size,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        (void**)&config));

        // Check if the kernel launch was successful.
        HIP_CHECK(hipGetLastError())

        return HIPRTC_SUCCESS;
    }

    int clear_run()
    {
        auto result = args.free();

        // Unload module.
        HIP_CHECK(hipModuleUnload(module));

        return result;
    }

private:
    std::string kernel_name;
    std::vector<char> code;
    std::string log;
    HipCodeArgs args;
    hipModule_t module;
};

static std::map<std::string, std::unique_ptr<HipCode>> codesByKernelName{};

int compile(const char* kernel_name, const char* kernel_code, const char** header_names, int header_names_count, const char** header_sources, int header_sources_count)
{
    codesByKernelName[kernel_name] = std::make_unique<HipCode>(kernel_name);
    auto &rtcClass = codesByKernelName[kernel_name];
    return rtcClass->compile(kernel_code, header_names, header_names_count, header_sources, header_sources_count);
}

size_t get_log(const char* kernel_name, char* log, size_t allocared_size)
{
    return codesByKernelName[kernel_name]->get_log(log, allocared_size);
}

size_t get_size(const char* kernel_name)
{
    return codesByKernelName[kernel_name]->get_size();
}

int param_int(const char* kernel_name, const char* param_name, int scalarValue)
{
    return codesByKernelName[kernel_name]->get_args().param(param_name, scalarValue);
}

int param_float(const char* kernel_name, const char* param_name, float scalarValue)
{
    return codesByKernelName[kernel_name]->get_args().param(param_name, scalarValue);
}

int param_float_array(const char* kernel_name, const char* param_name, const float* array_value, size_t count)
{
    return codesByKernelName[kernel_name]->get_args().param_array(param_name, array_value, count);
}

int run(const char* kernel_name, const char* function_name)
{
    return codesByKernelName[kernel_name]->run(function_name);
}

int result_float_array(const char* kernel_name, const char* param_name, float* return_float_array, size_t count)
{
    return codesByKernelName[kernel_name]->get_args().result_array(param_name, return_float_array, count);
}

int clear_run(const char* kernel_name)
{
    auto result = codesByKernelName[kernel_name]->clear_run();
    codesByKernelName.erase(kernel_name);
    return result;
}