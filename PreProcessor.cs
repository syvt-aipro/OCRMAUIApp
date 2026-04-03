using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

public class PreProcessor
{
    public static DenseTensor<float> PrepareDetectionTensor(Mat srcImg, int maxSideLen, out float ratioH, out float ratioW)
    {
        int w = srcImg.Cols;
        int h = srcImg.Rows;
        float ratio = 1.0f;
        int maxWh = Math.Max(w, h);
        
        if (maxWh > maxSideLen) ratio = (float)maxSideLen / maxWh;

        int resizeH = Math.Max(32, (int)(h * ratio / 32) * 32);
        int resizeW = Math.Max(32, (int)(w * ratio / 32) * 32);

        ratioH = (float)resizeH / h;
        ratioW = (float)resizeW / w;

        using Mat resizedImg = new Mat();
        Cv2.Resize(srcImg, resizedImg, new OpenCvSharp.Size(resizeW, resizeH));

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        return MatToTensor(resizedImg, mean, std);
    }

    public static DenseTensor<float> PrepareClassificationTensor(Mat srcImg)
    {
        using Mat resizedImg = new Mat();
        Cv2.Resize(srcImg, resizedImg, new OpenCvSharp.Size(160, 80));
        float[] mean = { 0.5f, 0.5f, 0.5f };
        float[] std = { 0.5f, 0.5f, 0.5f };
        return MatToTensor(resizedImg, mean, std);
    }

    public static DenseTensor<float> PrepareRecognitionTensor(Mat srcImg)
    {
        float ratio = (float)srcImg.Cols / srcImg.Rows;
        int resizeW = (int)Math.Ceiling(48 * ratio);
        resizeW = Math.Max(Math.Min(resizeW, 320), 48); // Max width 320

        using Mat resizedImg = new Mat();
        Cv2.Resize(srcImg, resizedImg, new OpenCvSharp.Size(resizeW, 48));
        
        // Pad to 320 if needed
        using Mat paddedImg = new Mat();
        Cv2.CopyMakeBorder(resizedImg, paddedImg, 0, 0, 0, 320 - resizeW, BorderTypes.Constant, Scalar.Black);

        float[] mean = { 0.5f, 0.5f, 0.5f };
        float[] std = { 0.5f, 0.5f, 0.5f };
        return MatToTensor(paddedImg, mean, std);
    }
    
    private static DenseTensor<float> MatToTensor(Mat img, float[] mean, float[] std)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, img.Rows, img.Cols });
        unsafe
        {
            byte* ptr = (byte*)img.DataPointer;
            int channels = img.Channels();
            int stride = (int)img.Step();

            for (int y = 0; y < img.Rows; y++)
            {
                for (int x = 0; x < img.Cols; x++)
                {
                    int offset = y * stride + x * channels;
                    float b = ptr[offset + 0] / 255.0f;
                    float g = ptr[offset + 1] / 255.0f;
                    float r = ptr[offset + 2] / 255.0f;

                    tensor[0, 0, y, x] = (r - mean[0]) / std[0];
                    tensor[0, 1, y, x] = (g - mean[1]) / std[1];
                    tensor[0, 2, y, x] = (b - mean[2]) / std[2];
                }
            }
        }
        return tensor;
    }

    public static Mat GetRotateCropImage(Mat src, RotatedRect rect)
    {
        Point2f[] pts = rect.Points();
        
        // Order points clockwise starting from top-left
        var orderedPts = pts.OrderBy(p => p.X).ToList();
        var lefts = orderedPts.Take(2).OrderBy(p => p.Y).ToList();
        var rights = orderedPts.Skip(2).OrderBy(p => p.Y).ToList();
        Point2f tl = lefts[0], bl = lefts[1], tr = rights[0], br = rights[1];

        int width = (int)Math.Max(tl.DistanceTo(tr), bl.DistanceTo(br));
        int height = (int)Math.Max(tl.DistanceTo(bl), tr.DistanceTo(br));

        Point2f[] srcPts = { tl, tr, br, bl };
        Point2f[] dstPts = { new Point2f(0, 0), new Point2f(width, 0), new Point2f(width, height), new Point2f(0, height) };

        using Mat m = Cv2.GetPerspectiveTransform(srcPts, dstPts);
        Mat dst = new Mat();
        Cv2.WarpPerspective(src, dst, m, new OpenCvSharp.Size(width, height), InterpolationFlags.Linear, BorderTypes.Replicate);

        if (dst.Rows >= dst.Cols * 1.5)
        {
            Cv2.Transpose(dst, dst);
            Cv2.Flip(dst, dst, FlipMode.X);
        }
        return dst;
    }
}