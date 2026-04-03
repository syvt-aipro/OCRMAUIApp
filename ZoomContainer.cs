using Microsoft.Maui.Controls;
using System;

namespace OCRMauiApp;

public class ZoomContainer : ContentView
{
    private double _currentScale = 1;
    private double _startScale = 1;
    private double _xOffset = 0;
    private double _yOffset = 0;

    public ZoomContainer()
    {
        IsClippedToBounds = true;
        
        // 1. Pinch to Zoom
        var pinchGesture = new PinchGestureRecognizer();
        pinchGesture.PinchUpdated += OnPinchUpdated;
        GestureRecognizers.Add(pinchGesture);

        // 2. Pan to Move
        var panGesture = new PanGestureRecognizer();
        panGesture.PanUpdated += OnPanUpdated;
        GestureRecognizers.Add(panGesture);

        // 3. Double Tap to Reset
        var doubleTapGesture = new TapGestureRecognizer { NumberOfTapsRequired = 2 };
        doubleTapGesture.Tapped += (s, e) => ResetZoom();
        GestureRecognizers.Add(doubleTapGesture);
    }

    private void OnPinchUpdated(object sender, PinchGestureUpdatedEventArgs e)
    {
        if (e.Status == GestureStatus.Started)
        {
            _startScale = Content.Scale;
            Content.AnchorX = 0.5;
            Content.AnchorY = 0.5;
        }
        if (e.Status == GestureStatus.Running)
        {
            // Calculate the scale and clamp it between 1x (normal) and 5x (max zoom)
            _currentScale += (e.Scale - 1) * _startScale;
            _currentScale = Math.Max(1, _currentScale);
            _currentScale = Math.Min(_currentScale, 5);

            Content.Scale = _currentScale;
        }
    }

    private void OnPanUpdated(object sender, PanUpdatedEventArgs e)
    {
        // Don't allow panning if the image isn't zoomed in
        if (Content.Scale <= 1) return; 

        if (e.StatusType == GestureStatus.Started)
        {
            _xOffset = Content.TranslationX;
            _yOffset = Content.TranslationY;
        }
        else if (e.StatusType == GestureStatus.Running)
        {
            // Move the image with the finger
            Content.TranslationX = _xOffset + e.TotalX;
            Content.TranslationY = _yOffset + e.TotalY;
        }
    }

    private void ResetZoom()
    {
        _currentScale = 1;
        Content.Scale = 1;
        Content.TranslationX = 0;
        Content.TranslationY = 0;
    }
}