import coremltools

output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
scale = 1/255.
coreml_model = coremltools.converters.keras.convert('./parameters.hdf5',
                                                   input_names       = 'image',
                                                   image_input_names = 'image',
                                                   output_names      = 'output',
                                                   class_labels      = output_labels,
                                                   image_scale       = scale)

coreml_model.author = 'Tobias Prisching'
coreml_model.license = 'No license'
coreml_model.short_description = 'Modell zur Klassifizierung von MNIST-Ziffern (CNN)'

coreml_model.input_description['image'] = 'Bild mit Ziffer (Graustufen, 28*28 Pixel)'
coreml_model.output_description['output'] = 'Erkannte Ziffer'

coreml_model.save('neural_net.mlmodel')
