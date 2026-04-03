using OpenCvSharp;
using Xamarin.Google.MLKit.Vision.Common;
using Android.Gms.Extensions;
using Xamarin.Google.MLKit.Vision.BarCode;
using Xamarin.Google.MLKit.Vision.Barcode.Common;

namespace OCRMauiApp;

public partial class MLKitScanner
{
    public partial async Task<(string text, List<OpenCvSharp.Rect> boxes)> ScanBarcodeAsync(string imagePath)
    {
        string barcodeText = "\n\n--- KẾT QUẢ MÃ VẠCH ---\n\n";
        List<OpenCvSharp.Rect> boundingBoxes = new List<OpenCvSharp.Rect>();

        try
        {
            var androidBitmap = Android.Graphics.BitmapFactory.DecodeFile(imagePath);
            var image = InputImage.FromBitmap(androidBitmap, 0);
            var options = new BarcodeScannerOptions.Builder()
                .SetBarcodeFormats(Barcode.FormatAllFormats)
                .Build();
            var scanner = BarcodeScanning.GetClient(options);
            var resultObj = await scanner.Process(image);
            
            var barcodes = new Android.Runtime.JavaList<Barcode>(resultObj.Handle, Android.Runtime.JniHandleOwnership.DoNotTransfer);

            if (barcodes.Count == 0) barcodeText += "Không tìm thấy mã vạch.\n";

            foreach (var barcode in barcodes)
            {
                barcodeText += $"• [{barcode.ValueType}] {barcode.RawValue}\n";
                if (barcode.BoundingBox != null)
                {
                    boundingBoxes.Add(new OpenCvSharp.Rect(
                        barcode.BoundingBox.Left, barcode.BoundingBox.Top, 
                        barcode.BoundingBox.Width(), barcode.BoundingBox.Height()));
                }
            }
        }
        catch (Exception ex) { barcodeText += $"Lỗi ML Kit: {ex.Message}\n"; }

        return (barcodeText, boundingBoxes);
    }
}