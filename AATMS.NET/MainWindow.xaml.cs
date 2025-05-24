using System;
using System.Windows;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Text;
using System.Collections.Generic;
using System.Linq;

namespace AATMS.NET
{
    public partial class MainWindow : Window
    {
        private string videoPath = string.Empty;
        private string modelPath = string.Empty;
        private string outputDirectory = string.Empty;
        private bool isProcessing = false;
        private string cppExecutablePath;

        public MainWindow()
        {
            InitializeComponent();
            InitializeCppBackend();
        }

        private void InitializeCppBackend()
        {
            try
            {
                // Set the path to the C++ executable
                cppExecutablePath = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory,
                    "aatms_cpp.exe"
                );

                if (!File.Exists(cppExecutablePath))
                {
                    StatusTextBlock.Text = "C++ backend executable not found. Please ensure aatms_cpp.exe is in the application directory.";
                    MessageBox.Show(
                        "C++ backend executable not found. Please ensure aatms_cpp.exe is in the application directory.",
                        "Error",
                        MessageBoxButton.OK,
                        MessageBoxImage.Error
                    );
                }
                else
                {
                    StatusTextBlock.Text = "C++ backend initialized successfully";
                }
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"Error initializing C++ backend: {ex.Message}";
                MessageBox.Show(
                    $"Error initializing C++ backend: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
            }
        }

        private void SelectVideo_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Video files (*.mp4)|*.mp4|All files (*.*)|*.*",
                Title = "Select Video File"
            };

            if (dialog.ShowDialog() == true)
            {
                videoPath = dialog.FileName;
                VideoPathTextBox.Text = videoPath;
            }
        }

        private void SelectModel_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "ONNX model files (*.onnx)|*.onnx|All files (*.*)|*.*",
                Title = "Select YOLO Model File"
            };

            if (dialog.ShowDialog() == true)
            {
                modelPath = dialog.FileName;
                ModelPathTextBox.Text = modelPath;
            }
        }

        private void SelectOutputDirectory_Click(object sender, RoutedEventArgs e)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select Output Directory"
            })
            {
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    outputDirectory = dialog.SelectedPath;
                    StatusTextBlock.Text = $"Output directory: {outputDirectory}";
                }
            }
        }

        private async void StartProcessing_Click(object sender, RoutedEventArgs e)
        {
            if (isProcessing)
            {
                StatusTextBlock.Text = "Processing is already in progress";
                return;
            }

            if (string.IsNullOrEmpty(videoPath) || string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(outputDirectory))
            {
                StatusTextBlock.Text = "Please select all required files and directories";
                MessageBox.Show(
                    "Please select all required files and directories",
                    "Missing Information",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning
                );
                return;
            }

            if (!File.Exists(cppExecutablePath))
            {
                StatusTextBlock.Text = "C++ backend executable not found";
                MessageBox.Show(
                    "C++ backend executable not found. Please ensure aatms_cpp.exe is in the application directory.",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
                return;
            }

            isProcessing = true;
            StatusTextBlock.Text = "Processing started...";

            try
            {
                string outputCsvPath = Path.Combine(
                    outputDirectory,
                    $"{Path.GetFileNameWithoutExtension(videoPath)}_detections.csv"
                );

                var startInfo = new ProcessStartInfo
                {
                    FileName = cppExecutablePath,
                    Arguments = $"--video \"{videoPath}\" --model \"{modelPath}\" --output \"{outputCsvPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = new Process { StartInfo = startInfo })
                {
                    var outputBuilder = new StringBuilder();
                    var errorBuilder = new StringBuilder();

                    process.OutputDataReceived += (s, args) =>
                    {
                        if (args.Data != null)
                        {
                            outputBuilder.AppendLine(args.Data);
                            Dispatcher.Invoke(() =>
                            {
                                StatusTextBlock.Text = args.Data;
                            });
                        }
                    };

                    process.ErrorDataReceived += (s, args) =>
                    {
                        if (args.Data != null)
                        {
                            errorBuilder.AppendLine(args.Data);
                        }
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    await Task.Run(() => process.WaitForExit());

                    if (process.ExitCode != 0)
                    {
                        throw new Exception($"C++ backend failed with exit code {process.ExitCode}. Error: {errorBuilder}");
                    }

                    if (File.Exists(outputCsvPath))
                    {
                        StatusTextBlock.Text = "Processing completed successfully";
                        MessageBox.Show(
                            $"Processing completed successfully!\nResults saved to: {outputCsvPath}",
                            "Success",
                            MessageBoxButton.OK,
                            MessageBoxImage.Information
                        );
                    }
                    else
                    {
                        throw new Exception("Output CSV file was not created");
                    }
                }
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"Error during processing: {ex.Message}";
                MessageBox.Show(
                    $"Error during processing: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
            }
            finally
            {
                isProcessing = false;
            }
        }
    }
} 