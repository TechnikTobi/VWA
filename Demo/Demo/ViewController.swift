//
//  ViewController.swift
//  Demo
//
//  Created by Tobias Prisching on 23.07.24.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    // UI variables
    @IBOutlet weak var uiDrawView:            DrawView!;
    @IBOutlet weak var uiImageView:           UIImageView!
    @IBOutlet weak var uiFFNNpredictionLabel: UILabel!
    @IBOutlet weak var uiCNNpredictionLabel:  UILabel!
    
    // Pixelbuffer stuff needed for converting the drawn image
    // into a value format that the neural network models understand
    var pixelBuffer: CVPixelBuffer?
    let pixelBufferAttributes = [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
    ] as CFDictionary;
    
    // Loading the neural network models
    let ffnn = FFNN();
    let cnn  = CNN();
    
    // Initial setup stuff
    override func
    viewDidLoad
    ()
    {
        super.viewDidLoad();
        
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            28,
            28,
            kCVPixelFormatType_OneComponent8,
            pixelBufferAttributes,
            &pixelBuffer
        );
        
        self.uiDrawView.viewController = self;
        
        self.uiFFNNpredictionLabel.isHidden = true;
        self.uiCNNpredictionLabel.isHidden  = true;
    }
    
    func
    classify
    ()
    {
        // Get the image from the DrawView
        let ciImage = self.uiDrawView.getCIimage();
        
        // Convert it to a pixelbuffer
        CIContext().render(ciImage!, to: self.pixelBuffer!);
        
        // Call the models to make their prediction
        let ffnn_output = try? ffnn.prediction(image: pixelBuffer!);
        let cnn_output  = try? cnn.prediction(image: pixelBuffer!);
        
        // Display predictions
        self.uiFFNNpredictionLabel.isHidden = false;
        self.uiCNNpredictionLabel.isHidden  = false;
        
        self.uiFFNNpredictionLabel.text = ffnn_output?.classLabel;
        self.uiCNNpredictionLabel.text  = cnn_output?.classLabel;
        
        let uiImage = UIImage(ciImage: ciImage!);
        uiImageView.image = uiImage;
    }
    
    @IBAction func
    uiClearButtonPressed
    (
        _ sender: Any
    )
    {
        self.uiDrawView.clear();
        
        self.uiFFNNpredictionLabel.isHidden = true;
        self.uiCNNpredictionLabel.isHidden  = true;
    }


}

