using ROCm.Hip;

namespace ROCm;

public class SaxpyHipProgram() : HipProgram(KernelName, src, HeaderNames, HeaderSources)
{
    // create program
    private static readonly string KernelName = "saxpy_kernel.cu";
    private static readonly string FunctionName = "saxpy_kernel";

    private static readonly string[] HeaderNames = { "header.h" };

    private static readonly string[] HeaderSources =
    {
        """
        #ifndef HIPRTC_HEADER_H
        #define HIPRTC_HEADER_H
        typedef float real;
        typedef float* realptr;
        typedef unsigned int uint;
        #define func extern "C" __global__ void
        #endif //HIPRTC_HEADER_H
        """
    };

    private static readonly string src = 
        """
        #include "header.h"

        func saxpy_kernel(const real a, const realptr d_x, realptr d_y, const uint size)
        {
            const uint global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(global_idx < size)
            {
                d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
            }
        }
        """;

    private float[]? _result;

    private float A
    {
        set => Param("a", value);
    }

    private float[] X
    {
        set => Param("d_x", value);
    }

    private float[] Y
    {
        set
        {
            Param("d_y", value);
            _result = value;
        }

        get
        {
            if (_result == null)
            {
                throw new NullReferenceException("d_y");
            }

            Out("d_y", _result);
            return _result;
        }
    }

    private int Size
    {
        set => Param("size", value);
    }


    public float[] Call(float a, float[] x, float[] y)
    {
        A = a;
        X = x;
        Y = y;
        Size = y.Length;

        Run(FunctionName);

        return Y;
    }
}