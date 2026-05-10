using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace PostureOSLauncher
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                // Set working directory to the folder where the .exe is
                string appDir = AppDomain.CurrentDomain.BaseDirectory;
                Directory.SetCurrentDirectory(appDir);

                // Path to the launcher bat
                string batPath = Path.Combine(appDir, "launchers", "windows", "PostureOS.bat");

                if (!File.Exists(batPath))
                {
                    MessageBox.Show(
                        "No se encontró el archivo de inicio en: " + batPath + "\n\n" +
                        "Asegúrate de haber extraído todos los archivos del ZIP.",
                        "PostureOS - Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return;
                }

                // Run the bat file
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = "cmd.exe";
                startInfo.Arguments = "/c \"" + batPath + "\"";
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = false; // Show the console for progress/errors

                Process.Start(startInfo);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error al iniciar PostureOS: " + ex.Message, "Error Fatal", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
