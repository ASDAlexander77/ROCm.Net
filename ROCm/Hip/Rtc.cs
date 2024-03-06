namespace ROCm.Hip;

using System.Runtime.InteropServices;
using System.Text;

public enum Result
{
    /// Success
    Success = 0,

    /// Out of memory
    ErrorOutOfMemory = 1,

    /// Failed to create program
    ErrorProgramCreationFailure = 2,

    /// Invalid input
    ErrorInvalidInput = 3,

    /// Invalid program
    ErrorInvalidProgram = 4,

    /// Invalid option
    ErrorInvalidOption = 5,

    /// Compilation error
    ErrorCompilation = 6,

    /// Failed in builtin operation
    ErrorBuiltinOperationFailure = 7,

    /// No name expression after compilation
    ErrorNoNameExpressionsAfterCompilation = 8,

    /// No lowered names before compilation 
    ErrorNoLoweredNamesBeforeCompilation = 9,

    /// Invalid name expression
    ErrorNameExpressionNotValid = 10,

    /// Internal error
    ErrorInternalError = 11,

    /// Error in linking
    ErrorLinking = 100 
};

public static class Rtc
{
    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "compile", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result Compile(
        string kernelName,
        string src,
        string[] headerNames,
        int headerNamesCount,
        string[] headerSources,
        int headerSourcesCount);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "get_log", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int Log(
        string kernelName,
        StringBuilder? log,
        int allocatedSize);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "get_size", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result CodeSize(
        string kernelName);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "param_int", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result ParamInt32(
        string kernelName,
        string paramName,
        int value);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "param_float", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result ParamFloat(
        string kernelName,
        string paramName,
        float value);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "param_float_array", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result ParamFloatArray(
        string kernelName,
        string paramName,
        float[] value, 
        int count);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "run", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result Run(
        string kernelName, 
        string functionName);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "result_float_array", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result ResultFloatArray(
        string kernelName,
        string paramName,
        float[] value, 
        int count);

    [DllImport("shared_hip_runtime_compilation_shared_vs2022", EntryPoint = "clear_run", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern Result CleanRun(
        string kernelName);
}