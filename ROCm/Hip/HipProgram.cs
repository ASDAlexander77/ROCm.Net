using System.Runtime.InteropServices;
using System.Text;

namespace ROCm.Hip;

public class HipProgram(string kernelName, string src, string[] headerNames, string[] headerSources) : IDisposable
{
    private bool isDisposed;
    private Result state;

    public bool IsCompiled { get; private set; }

    public string Log
    {
        get
        {
            var size = Rtc.Log(kernelName, null, 0);
            var log = new StringBuilder(size);
            Rtc.Log(kernelName, log, size);
            return log.ToString();
        }
    }

    public void Dispose()
    {
        if (isDisposed) return;
        isDisposed = true;
        Rtc.CleanRun(kernelName);
    }

    public void Compile()
    {
        Check(Rtc.Compile(kernelName, src, headerNames, headerNames.Length, headerSources, headerSources.Length));
        IsCompiled = true;
    }

    public float[] Out(string paramName, [Out] float[] result)
    {
        Rtc.ResultFloatArray(kernelName, "d_y", result, result.Length);
        return result;
    }

    public void Param(string paramName, float value)
    {
        Check(Rtc.ParamFloat(kernelName, paramName, value));
    }

    public void Param(string paramName, int value)
    {
        Check(Rtc.ParamInt32(kernelName, paramName, value));
    }

    public void Param(string paramName, float[] value)
    {
        Check(Rtc.ParamFloatArray(kernelName, paramName, value, value.Length));
    }

    public void Run(string functionName)
    {
        Check(Rtc.Run(kernelName, functionName));
    }

    private void Check(Result checkState)
    {
        state = checkState;
        if (state != Result.Success)
        {
            throw new InvalidOperationException(state.ToString());
        }
    }
}