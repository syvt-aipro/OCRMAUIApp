using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

public class OcrPipeline
{
    private readonly InferenceSession _detSession;
    private readonly InferenceSession _clsSession;
    private readonly InferenceSession _recSession;
    private readonly string[] _dictionary;

    public OcrPipeline(string detModelPath, string clsModelPath, string recModelPath, string dictPath)
    {
        SessionOptions options = new SessionOptions();
        options.AppendExecutionProvider_CPU(); 
        
        // Load sessions from file paths to support models with external .data weights
        _detSession = new InferenceSession(detModelPath, options);
        _clsSession = new InferenceSession(clsModelPath, options);
        _recSession = new InferenceSession(recModelPath, options);

        // Load dictionary and prepend blank space for CTC decoding
        var dictLines = File.ReadAllLines(dictPath).ToList();
        dictLines.Insert(0, "#"); // Blank token
        dictLines.Add(" ");       // Space token
        _dictionary = dictLines.ToArray();
    }

    public OcrResult ProcessImage(Mat srcImg)
    {
        List<string> recognizedTexts = new List<string>();
        
        using Mat annotatedImg = srcImg.Clone();

        // 1. Detection
        var detTensor = PreProcessor.PrepareDetectionTensor(srcImg, 960, out float ratioH, out float ratioW);
        var detInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", detTensor) };
        using var detResults = _detSession.Run(detInputs);
        var predData = detResults.First().AsTensor<float>().ToArray();
        
        // Extract Boxes and scale them back to original image size
        var boxes = DbPostProcessor.ExtractBoundingBoxes(predData, detTensor.Dimensions[3], detTensor.Dimensions[2]);
        
        foreach (var box in boxes)
        {
            // Scale box to original image
            var scaledCenter = new Point2f(box.Center.X / ratioW, box.Center.Y / ratioH);
            var scaledSize = new Size2f(box.Size.Width / ratioW, box.Size.Height / ratioH);
            var scaledBox = new RotatedRect(scaledCenter, scaledSize, box.Angle);
            
            var points = scaledBox.Points().Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray();
            Cv2.Polylines(annotatedImg, new[] { points }, isClosed: true, color: Scalar.Green, thickness: 2);

            // 2. Crop
            using Mat cropImg = PreProcessor.GetRotateCropImage(srcImg, scaledBox);
            if (cropImg.Empty()) continue;

            // 3. Direction Classification
            var clsTensor = PreProcessor.PrepareClassificationTensor(cropImg);
            var clsInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", clsTensor) };
            using var clsResults = _clsSession.Run(clsInputs);
            var clsScores = clsResults.First().AsTensor<float>().ToArray();
            
            // If label is 1 (180 degrees) and score > 0.9, flip the image
            if (clsScores[1] > clsScores[0] && clsScores[1] > 0.9f)
            {
                Cv2.Rotate(cropImg, cropImg, RotateFlags.Rotate180);
            }

            // 4. Recognition
            var recTensor = PreProcessor.PrepareRecognitionTensor(cropImg);
            var recInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", recTensor) };
            using var recResults = _recSession.Run(recInputs);
            var recData = recResults.First().AsTensor<float>().ToArray();

            // Decode CTC
            string text = DbPostProcessor.DecodeCTC(recData, _dictionary);
            if (!string.IsNullOrWhiteSpace(text))
            {
                recognizedTexts.Add(text);
            }
        }
        byte[] imgBytes = annotatedImg.ImEncode(".jpg");

        return new OcrResult 
        { 
            Texts = recognizedTexts, 
            AnnotatedImageBytes = imgBytes 
        };
    }
}

public class OcrResult
{
    public List<string> Texts { get; set; } = new List<string>();
    public byte[]? AnnotatedImageBytes { get; set; }
}