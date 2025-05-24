using System;
using System.Diagnostics;
using System.Text;
using System.IO;

namespace DotNetPythonIntegration
{
    class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                // Get the current directory where the executable is located
                string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
                string pythonScriptPath = Path.Combine(currentDirectory, "python_script.py");

                // Verify Python script exists
                if (!File.Exists(pythonScriptPath))
                {
                    Console.WriteLine($"Error: Python script not found at {pythonScriptPath}");
                    return;
                }

                // Create process start info
                var start = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScriptPath}\" 5 3",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                // Start the process
                using var process = Process.Start(start);
                if (process == null)
                {
                    Console.WriteLine("Error: Failed to start Python process");
                    return;
                }

                // Read the output
                string result = await process.StandardOutput.ReadToEndAsync();
                string error = await process.StandardError.ReadToEndAsync();

                // Wait for the process to exit
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    Console.WriteLine($"Python script result: {result.Trim()}");
                }
                else
                {
                    Console.WriteLine($"Error running Python script: {error.Trim()}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
} 