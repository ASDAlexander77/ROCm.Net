// See https://aka.ms/new-console-template for more information

using ROCm;

Console.WriteLine($"Hello, ROCm!");

using var hipProg = new SaxpyHipProgram();
try
{
    hipProg.Compile();
    if (hipProg.IsCompiled)
    {
        var result = hipProg.Call(
            3.0f, 
            new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 
            new float[] { -1, -2, -3, -4, -5, -6, -7, -8 });

        foreach (var value in result)
        {
            Console.Write(value);
            Console.Write(" ");
        }

        Console.WriteLine(string.Empty);

        var result2 = hipProg.Call(
            4.0f,
            new float[] { 8, 7, 6, 5, 4, 3, 2, 1 },
            new float[] { -8, -7, -6, -5, -4, -3, -2, -1 });

        foreach (var value in result2)
        {
            Console.Write(value);
            Console.Write(" ");
        }

        Console.WriteLine(string.Empty);
    }
    else
    {
        Console.WriteLine(hipProg.Log);
    }
}
catch (Exception e)
{
    Console.WriteLine(e);
}

Console.WriteLine("Cool, it worked");
