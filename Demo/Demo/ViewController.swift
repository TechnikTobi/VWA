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
    @IBOutlet weak var uiVWApredictionLabel:  UILabel!
    @IBOutlet weak var uiExt1predictionLabel: UILabel!
    @IBOutlet weak var uiExt2predictionLabel: UILabel!
    
    // Pixelbuffer stuff needed for converting the drawn image
    // into a value format that the neural network models understand
    var pixelBuffer: CVPixelBuffer?
    let pixelBufferAttributes = [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
    ] as CFDictionary;
    
    // Loading the neural network models
    
    // Old models, not used during presentation:
    // let ffnn = FFNN();
    // let cnn  = CNN();
    
    // Model that stems from the VWA research:
    let vwa_model  = CNN4softmax();
    
    // External model Nr. 1, based on:
    // https://github.com/keras-team/keras/blob/4f2e65c385d60fa87bb143c6c506cbe428895f44/examples/mnist_cnn.py
    let ext1_model = keras_mnist_cnn();
    
    // External model Nr. 2, based on:
    //
    let ext2_model = mnistCNN();
    
    // 138-140: nicht zu gebrauchen
    // 141: ganz gut (zw. CNN3 und mnistCNN)
    // CNN4sigmoid minimal besser als CNN3
    // CNN4softmax besser als CNN5epoch5 (CNN4 is eigentlich bestes aus eigener CNN Reihe bisher)
    
    
    // Initial setup stuff
    override func
    viewDidLoad
    ()
    {
        super.viewDidLoad();
     
        // Set up the pixelbuffer
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            28,
            28,
            kCVPixelFormatType_OneComponent8,
            pixelBufferAttributes,
            &pixelBuffer
        );
        
        self.uiDrawView.viewController = self;
        
        // Disable interpolation
        self.uiImageView.layer.magnificationFilter = .nearest;
        
        // Hide the prediction text labels for now
        self.setPredictionLabelsAreHidden(true);
        
        // First render to image view
        self.showDrawViewInImageView();
    }
    
    func
    classify
    ()
    {
        // Get the image from the DrawView
        let ciImage = self.uiDrawView.getCIimage();
        
        // Show the image in the image view
        let uiImage = UIImage(ciImage: ciImage!);
        uiImageView.image = uiImage;
        
        // Convert it to a pixelbuffer
        CIContext().render(ciImage!, to: self.pixelBuffer!);
        
        // Call the models to make their prediction
        let vwa_output  = try? self.vwa_model.prediction(image: pixelBuffer!);
        let ext1_output = try? self.ext1_model.prediction(image: pixelBuffer!);
        let ext2_output = try? self.ext2_model.prediction(image: pixelBuffer!);
        
        // Display predictions
        self.setPredictionLabelsAreHidden(false);
        
        self.uiVWApredictionLabel.text  = vwa_output?.classLabel;
        self.uiExt1predictionLabel.text = ext1_output?.classLabel;
        self.uiExt2predictionLabel.text = ext2_output?.classLabel;
    }
    
    func
    showDrawViewInImageView
    ()
    {
        uiImageView.image = UIImage(
            ciImage: self.uiDrawView.getCIimage()!
        );
    }
    
    func
    setPredictionLabelsAreHidden
    (
        _ value: Bool
    )
    {
        self.uiVWApredictionLabel.isHidden  = value;
        self.uiExt1predictionLabel.isHidden = value;
        self.uiExt2predictionLabel.isHidden = value;
    }
    
    @IBAction func
    uiClearButtonPressed
    (
        _ sender: Any
    )
    {
        // Clear the draw view, image view and hide text labels
        self.uiDrawView.clear();
        self.showDrawViewInImageView();
        self.setPredictionLabelsAreHidden(true);
    }


}

