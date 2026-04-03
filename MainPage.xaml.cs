﻿using Microsoft.Maui.Media;
using OpenCvSharp;

namespace OCRMauiApp;

public partial class MainPage : ContentPage
{
    private string _photoPath;
    private OcrPipeline _ocrPipeline;

    public MainPage()
    {
        InitializeComponent();
    }
    
    // 1. Handle Camera Click
    private async void OnCaptureClicked(object sender, EventArgs e)
    {
        if (MediaPicker.Default.IsCaptureSupported)
        {
            var photo = await MediaPicker.Default.CapturePhotoAsync();
            await ProcessPhotoAsync(photo);
        }
    }

    // 2. Handle Gallery Click
    private async void OnGalleryClicked(object sender, EventArgs e)
    {
        var photo = await MediaPicker.Default.PickPhotoAsync();
        await ProcessPhotoAsync(photo);
    }

    // 3. Shared Logic to save and display the image
    private async Task ProcessPhotoAsync(FileResult photo)
    {
        if (photo != null)
        {
            // Save to app cache
            var localFile = Path.Combine(FileSystem.CacheDirectory, photo.FileName);
            using var stream = await photo.OpenReadAsync();
            using var newStream = File.OpenWrite(localFile);
            await stream.CopyToAsync(newStream);

            _photoPath = localFile;
            ResultImage.Source = ImageSource.FromFile(_photoPath);
            
            // Enable the OCR button and reset text
            BtnRunOcr.IsEnabled = true;
            TvResult.Text = "Đã tải ảnh. Sẵn sàng chạy OCR.";
        }
    }
    
    // Helper to extract Raw assets to the local file system
    private async Task<string> ExtractAssetAsync(string filename)
    {
        var localPath = Path.Combine(FileSystem.CacheDirectory, filename);
        if (!File.Exists(localPath))
        {
            using var stream = await FileSystem.OpenAppPackageFileAsync(filename);
            using var newStream = File.OpenWrite(localPath);
            await stream.CopyToAsync(newStream);
        }
        return localPath;
    }

    private async void OnRunOcrClicked(object sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_photoPath)) return;

        BtnRunOcr.IsEnabled = false;
        TvResult.Text = "Đang đọc chữ và mã vạch...";

        await Task.Run(async () =>
        {
            try
            {
                // 1. Extract all required files to the physical cache directory
                string detPath = await ExtractAssetAsync("PP-OCRv5_mobile_det.onnx");
                string recPath = await ExtractAssetAsync("PP-OCRv5_mobile_rec.onnx");
                string clsPath = await ExtractAssetAsync("PP-LCNet_x0_25_textline_ori.onnx");
                string dictPath = await ExtractAssetAsync("ppocr_keys_ocrv5.txt");
                
                // If you use ESRGAN, extract BOTH the .onnx and the .data files
                await ExtractAssetAsync("real_esrgan_general_x4v3.onnx");
                await ExtractAssetAsync("real_esrgan_general_x4v3.data");

                // 2. Initialize the pipeline (only do this once in a real app to save memory)
                _ocrPipeline ??= new OcrPipeline(detPath, clsPath, recPath, dictPath);
                
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                // 3. Run the full OCR process
                using Mat srcImg = Cv2.ImRead(_photoPath, ImreadModes.Color);
                var ocrResult = _ocrPipeline.ProcessImage(srcImg);
                
                // 4. Run ML Kit Barcode Scanning
                var mlKitScanner = new MLKitScanner();
                var (barcodeText, barcodeBoxes) = await mlKitScanner.ScanBarcodeAsync(_photoPath);
                
                stopwatch.Stop();
                double elapsedSeconds = stopwatch.ElapsedMilliseconds / 1000.0;
                
                // 5. Draw the Red ML Kit bounding boxes using OpenCV
                using Mat finalImg = Cv2.ImDecode(ocrResult.AnnotatedImageBytes, ImreadModes.Color);
                foreach (var box in barcodeBoxes)
                {
                    Cv2.Rectangle(finalImg, box, Scalar.Red, thickness: 4);
                }
                byte[] finalImgBytes = finalImg.ImEncode(".jpg");

                // 6. Format Output
                string finalText = string.Join("\n", ocrResult.Texts) + barcodeText;

                MainThread.BeginInvokeOnMainThread(() =>
                {
                    TvResult.Text = $"--- KẾT QUẢ ĐỌC CHỮ ---\n{finalText}";
                    ResultImage.Source = ImageSource.FromStream(() => new MemoryStream(finalImgBytes));
                    BtnRunOcr.IsEnabled = true;
                    
                    BtnRunOcr.Text = $"Chạy lại OCR ({elapsedSeconds:F3} s)";
                    BtnRunOcr.IsEnabled = true;
                });
            }
            catch (Exception ex)
            {
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    TvResult.Text = $"AI Error: {ex.Message}";
                    BtnRunOcr.IsEnabled = true;
                    
                    BtnRunOcr.Text = "Chạy lại OCR";
                    BtnRunOcr.IsEnabled = true;
                });
            }
        });
    }
}