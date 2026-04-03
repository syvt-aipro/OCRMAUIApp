using OpenCvSharp;
using UIKit;
using MLKit.Core;
using MLKit.BarcodeScanning;

namespace OCRMauiApp;

public partial class MLKitScanner
{
    public partial async Task<(string text, List<OpenCvSharp.Rect> boxes)> ScanBarcodeAsync(string imagePath)
    {
        string barcodeText = "\n\n--- KẾT QUẢ MÃ VẠCH ---\n\n";
        List<OpenCvSharp.Rect> boundingBoxes = new List<OpenCvSharp.Rect>();

        try
        {
            // Load iOS UIImage
            var uiImage = UIImage.FromFile(imagePath);
            if (uiImage == null) return ("Lỗi: Không thể đọc file ảnh iOS.", boundingBoxes);
            
            var visionImage = new MLKVisionImage(uiImage);
            visionImage.Orientation = uiImage.Orientation;
            
            var format = new MLKBarcodeScannerOptions();
            var scanner = MLKBarcodeScanner.BarcodeScannerWithOptions(format);

            // Execute Apple/Swift ML Kit Task
            MLKBarcode[] barcodes = await scanner.ProcessImageAsync(visionImage);

            if (barcodes == null || barcodes.Length == 0) barcodeText += "Không tìm thấy mã vạch.\n";

            foreach (var barcode in barcodes)
            {
                barcodeText += $"• [{barcode.ValueType}] {barcode.RawValue}\n";
                
                // Extract CoreGraphics Rect and convert to OpenCvSharp Rect
                var cgRect = barcode.Frame;
                boundingBoxes.Add(new OpenCvSharp.Rect(
                    (int)cgRect.X, (int)cgRect.Y, 
                    (int)cgRect.Width, (int)cgRect.Height));
            }
        }
        catch (Exception ex) { barcodeText += $"Lỗi ML Kit: {ex.Message}\n"; }

        return (barcodeText, boundingBoxes);
    }
}