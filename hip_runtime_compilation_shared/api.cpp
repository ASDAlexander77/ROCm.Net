#include "api.h"

#include "utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <string>
#include <vector>
#include <map>

struct HipCodeArgs
{
public:
    HipCodeArgs() : paramNames{}, offset{0}, size{0}, args{0}, allocated{} {};

    template <typename T>
    int param(std::string param_name, T scalarValue)
    {
        auto deltaOffset = std::max(sizeof(scalarValue), sizeof(void*)); // aligning fix for CUDA executions

        if (offset + deltaOffset > sizeof(args))
        {
            return -1;
        }

        *(reinterpret_cast<T*>(&(args[offset]))) = scalarValue;
        offset += deltaOffset;

        paramNames.push_back(param_name);

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
        auto deltaOffset = sizeof(d_arrayValue);

        if (offset + deltaOffset > sizeof(args))
        {
            return -1;
        }

        HIP_CHECK(hipMalloc(&d_arrayValue, size_bytes));

        allocated.push_back((void*)d_arrayValue);

        HIP_CHECK(hipMemcpy(d_arrayValue, array_value, size_bytes, hipMemcpyHostToDevice));

        *(reinterpret_cast<const T**>(&(args[offset]))) = d_arrayValue;
        offset += deltaOffset;

        paramNames.push_back(param_name);

        return HIPRTC_SUCCESS;
    }

    template <typename T>
    int result_array(std::string param_name, T** return_array, size_t count)
    {
        auto args_as_ptrs = reinterpret_cast<void**>(&args);

        // size for Float array
        size_t size_bytes = size * sizeof(T);

        auto it = std::find(paramNames.begin(), paramNames.end(), param_name);
        if (it == std::end(paramNames))
        {
            return -1;
        }

        auto param_index = std::distance(paramNames.begin(), it);

        auto d_pointer = args_as_ptrs[param_index];

        // Copy results from device to host.
        HIP_CHECK(hipMemcpy(return_array, d_pointer, size_bytes, hipMemcpyDeviceToHost));

        return HIPRTC_SUCCESS;
    }

    int free()
    {
        // Free device memory.
        for (auto& ptr : allocated)
        {
            HIP_CHECK(hipFree((float*)ptr));
        }

        return HIPRTC_SUCCESS;
    }

    size_t offset;
    size_t size;
    char args[256];

private:
    std::vector<std::string> paramNames;
    std::vector<void*> allocated;
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
            log.reserve(log_size);
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
        code.reserve(code_size);
        hiprtcGetCode(prog, code.data());

        // Destroy program object.
        hiprtcDestroyProgram(&prog);

        return HIPRTC_SUCCESS;
    }

    size_t get_log(char* log_out, size_t allocared_size)
    {
        if (log.size() <= allocared_size)
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
        auto offset = args.offset;
        // Total number of float elements in each device vector.
        auto size = args.size;
        auto &argsRef = args.args;

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
                          argsRef,
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

int result_float_array(const char* kernel_name, const char* param_name, float** return_float_array, size_t count)
{
    return codesByKernelName[kernel_name]->get_args().result_array(param_name, return_float_array, count);
}

int clear_run(const char* kernel_name)
{
    auto result = codesByKernelName[kernel_name]->clear_run();
    codesByKernelName.erase(kernel_name);
    return result;
}