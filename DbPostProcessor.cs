using OpenCvSharp;
using Clipper2Lib;
using System.Text;

public class DbPostProcessor
{
    public static List<RotatedRect> ExtractBoundingBoxes(float[] predData, int width, int height, double boxThresh = 0.5, double unclipRatio = 1.6)
    {
        using Mat predMap = new Mat(height, width, MatType.CV_32FC1);
        predMap.SetArray(predData);
        
        using Mat bitMap = new Mat();
        using Mat cbufMap = new Mat();
        predMap.ConvertTo(cbufMap, MatType.CV_8UC1, 255.0);
        Cv2.Threshold(cbufMap, bitMap, boxThresh * 255.0, 255.0, ThresholdTypes.Binary);

        using Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(2, 2));
        Cv2.Dilate(bitMap, bitMap, kernel);

        Cv2.FindContours(bitMap, out OpenCvSharp.Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        List<RotatedRect> boxes = new List<RotatedRect>();

        foreach (var contour in contours)
        {
            if (contour.Length <= 2) continue;

            RotatedRect box = Cv2.MinAreaRect(contour);
            float ssid = Math.Min(box.Size.Width, box.Size.Height);
            if (ssid < 3) continue;

            // 1. Calculate Unclip Distance (area * unclip_ratio / perimeter)
            double area = Cv2.ContourArea(contour);
            double length = Cv2.ArcLength(contour, true);
            double distance = area * unclipRatio / length;

            // 2. Expand Polygon using Clipper2
            Path64 path = new Path64();
            foreach (var pt in contour) path.Add(new Point64(pt.X, pt.Y));
            Paths64 paths = new Paths64 { path };

            Paths64 offsetPaths = Clipper.InflatePaths(paths, distance, JoinType.Round, EndType.Polygon);
            
            if (offsetPaths.Count == 0) continue;

            // 3. Convert back to RotatedRect
            var inflatedPoints = offsetPaths[0].Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray();
            RotatedRect unclippedBox = Cv2.MinAreaRect(inflatedPoints);

            if (Math.Min(unclippedBox.Size.Width, unclippedBox.Size.Height) >= 3)
            {
                boxes.Add(unclippedBox);
            }
        }

        // Sort boxes from top to bottom
        return boxes.OrderBy(b => b.Center.Y).ToList();
    }

    // CTC Decoding for Recognition output
    public static string DecodeCTC(float[] recData, string[] dictionary)
    {
        int dictSize = dictionary.Length;
        int timeSteps = recData.Length / dictSize;
        
        StringBuilder sb = new StringBuilder();
        int lastIndex = 0;

        for (int t = 0; t < timeSteps; t++)
        {
            int maxIdx = 0;
            float maxVal = 0;

            for (int i = 0; i < dictSize; i++)
            {
                float val = recData[t * dictSize + i];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = i;
                }
            }

            // Ignore blank tokens (0) and repeated consecutive characters
            if (maxIdx > 0 && !(t > 0 && maxIdx == lastIndex))
            {
                sb.Append(dictionary[maxIdx]);
            }
            lastIndex = maxIdx;
        }

        return sb.ToString();
    }
}