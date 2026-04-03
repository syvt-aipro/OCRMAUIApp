using OpenCvSharp;

namespace OCRMauiApp;

public partial class MLKitScanner
{
    // This is just a definition. The actual code will be written in the Platforms folders.
    public partial Task<(string text, List<OpenCvSharp.Rect> boxes)> ScanBarcodeAsync(string imagePath);
}