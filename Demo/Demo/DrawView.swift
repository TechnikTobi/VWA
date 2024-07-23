//
//  DrawView.swift
//  Demo
//
//  Created by Tobias Prisching on 23.07.24.
//

import Foundation
import UIKit

class DrawView: UIView
{
    let linewidth = CGFloat(55);
    let color     = UIColor.white;
    
    var lastPoint: CGPoint!;
    var lines: [(CGPoint, CGPoint)] = [];    
    
    @IBInspectable weak var viewController: ViewController?
    
    override func
    touchesBegan
    (
        _ touches: Set<UITouch>, 
        with event: UIEvent?
    )
    {
        self.lastPoint = touches.first!.location(in: self);
    }
    
    
    
    override func
    touchesMoved
    (
        _ touches: Set<UITouch>,
        with event: UIEvent?
    )
    {
        let nextPoint = touches.first!.location(in: self);
        
        lines.append((lastPoint, nextPoint));
        lastPoint = nextPoint;
        
        // Tells the system that the view needs to be (re)drawn
        setNeedsDisplay();
    }
    
    
    
    override func
    touchesEnded
    (
        _ touches: Set<UITouch>,
        with event: UIEvent?
    )
    {
        self.viewController!.classify();
    }
    
    
    
    override func
    draw
    (
        _ rect: CGRect
    )
    {
        super.draw(rect);
        
        let drawPath = UIBezierPath();
        drawPath.lineCapStyle = .round;
        
        for line in self.lines
        {
            drawPath.move(to: line.0);
            drawPath.addLine(to: line.1);
        }
        
        // Tell the path how wide to be drawn
        drawPath.lineWidth = self.linewidth;
        
        // Set drawing color
        color.set();
        
        // Perform the function call for actually drawing the path
        drawPath.stroke();
    }
    
    
    
    func
    getCGcontext
    ()
    -> CGContext?
    {
        let colorSpace = CGColorSpaceCreateDeviceGray();
        let bitmapInfo = CGImageAlphaInfo.none.rawValue;
        
        let context = CGContext(
            data: nil,
            width: 28, height: 28,
            bitsPerComponent: 8, bytesPerRow: 28,
            space: colorSpace, bitmapInfo: bitmapInfo
        );
        
        context?.translateBy(x: 0, y: 28);
        context?.scaleBy(
            x:  28/self.frame.size.width,
            y: -28/self.frame.size.height
        );

        self.layer.render(in: context!);
        
        return context
    }
    
    
    
    func
    getCIimage
    ()
    -> CIImage?
    {
        let cgImage = self.getCGcontext()?.makeImage();
        return CIImage(cgImage: cgImage!);
    }
    
    
    
    func
    clear
    ()
    {
        self.lines = [];
        setNeedsDisplay();
    }
}
