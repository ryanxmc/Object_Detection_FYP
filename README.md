# Object_Detection_FYP
This repository contains all the files needed to run our object detection model - which can take a small input of images of ovals and stars, along with their labels in XML format, to create an accurate and robust object detection model.

I have included the images and labels we used, as well as necessary files needed to run the model.

Code is well commented to ensure that it can be easily replicated.

For convenience, create a folder named "2CustomTF2" to work from and set up your directory like so:

    > 2CustomTF2
  
      > data
    
        > images (jpg files provided)
        > annotations (xml files provided)
        > test_labels (place 3 of the xml files at random in here)
        > train_labels (place the remaining 9 xml files in here)
      
      > training
      
Everything else is well guided, however for reference, just before you start training your directory should look like so:

    > 2CustomTF2
  
      > data
    
        > images (jpg files provided)
        > annotations (xml files provided)
        > test_labels (place 3 of the xml files at random in here)
        > train_labels (place the remaining 9 xml files in here)
        > label_map.pbtxt
        > mobilenet_v2.config
        > test.record
        > train.record
        > test_labels.csv
        > train_labels.csv
      
      > training
      
      > models
      
      > mobilenet_v2
      
      > generate_tfrecord.py
      
      > mobilenet_v2.config


Finally, after training, you will be able to view the results via TensorBoard
