```python
# importing the required libraries for object detection

import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
```


```python
### the following block is not needed if you are working locally - this was for use in Google Colab

# setting up goole drive link

from google.colab import drive
drive.mount('/content/gdrive')

# this creates a symbolic link so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
print(tf.__version__)
```


```python
### Create a folder "CustomTF2" and set up your directory as follows:

# CustomTF2
    # data
        # images (JPG images we have provided)
        # annotations (XML files we have provided)
        # test_labels (25% of the XML files, select at random)
        # train_labels (75% of the XML files, select at random)
    
    
    # training
    
### Upload the generate_tfrecord.py file from github and place into the CustomTF2 folder
```


```python
## Install TF Object Detection API and clone repositorys from github using the following code below
```


```python
# clone the tensorflow models on the colab cloud vm
!git clone --q https://github.com/tensorflow/models.git
```


```python
#navigate to /models/research folder to compile protos
os.chdir("/content/gdrive/MyDrive/CustomTF2/models/research") # may need to alter path before /CustomTF2

# Compile protos.
!protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```


```python
# testing the model builder - filepath may need edited here
!python /content/gdrive/MyDrive/2CustomTF2/models/research/object_detection/builders/model_builder_tf2_test.py
```


```python
### We now need to change the XML files into CSV format, so that they can then be easily transferred to .record files

# Change working directory to CustomTF2/data

os.chdir("/content/gdrive/MyDrive/2CustomTF2/data") # path to /data folder

# This will also create the label_map.pbtxt file
# This is used in the models .config file

#adjusted from: https://github.com/datitran/raccoon_dataset
def xml_to_csv(path):
  classes_names = []
  xml_list = []

  for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
      classes_names.append(member[0].text)
      value = (root.find('filename').text  ,   
               int(root.find('size')[0].text),
               int(root.find('size')[1].text),
               member[0].text,
               int(member[4][0].text),
               int(member[4][1].text),
               int(member[4][2].text),
               int(member[4][3].text))
      xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name) 
  classes_names = list(set(classes_names))
  classes_names.sort()
  return xml_df, classes_names

for label_path in ['train_labels', 'test_labels']:
  image_path = os.path.join(os.getcwd(), label_path)
  xml_df, classes = xml_to_csv(label_path)
  xml_df.to_csv(f'{label_path}.csv', index=None)
  print(f'Successfully converted {label_path} xml to csv.')

label_map_path = os.path.join("label_map.pbtxt")
pbtxt_content = ""

for i, class_name in enumerate(classes):
    pbtxt_content = (
        pbtxt_content
        + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, class_name)
    )
pbtxt_content = pbtxt_content.strip()
with open(label_map_path, "w") as f:
    f.write(pbtxt_content)
    print('Successfully created label_map.pbtxt ')   


```


```python
# Our CustomTF2 directory should now also contain a label_map.pbtxt file, as well as csv files.
```


```python
## Next step is to create .record files from these csv csv files

# Ensure current wd is /data

os.chdir("/content/gdrive/MyDrive/2CustomTF2/data") # change accordingly

# Ensure to change file paths to the generate_tfrecord file that we downloaded

#For train.record
!python /content/gdrive/MyDrive/2CustomTF2/generate_tfrecord.py train_labels.csv  label_map.pbtxt images/ train.record

#For test.record
!python /content/gdrive/MyDrive/2CustomTF2/generate_tfrecord.py test_labels.csv  label_map.pbtxt images/ test.record

```


```python
# Now to download our model, SSD_MobileNet_v2 
os.chdir('/content/gdrive/MyDrive/2CustomTF2') # 2CustomTF2 wd
!wget http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilenet_v2.tar.gz
!tar -xvf mobilenet_v2.tar.gz
!rm mobilenet_v2.tar.gz

!wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config
!mv ssd_mobilenet_v2_320x320_coco17_tpu-8.config mobilenet_v2.config
```


```python
# This next box of code is flexible in batch size, num_steps and num_eval_steps
# I have kept the settings we settled on


# Important box of code
num_classes = 2 # star, oval
batch_size = 64
num_steps = 7500
num_eval_steps = 1000

# these all need adjusted to suit your own directory setup

# path to train.record
train_record_path = '/content/gdrive/MyDrive/2CustomTF2/data/train.record'

# path to test.record
test_record_path = '/content/gdrive/MyDrive/2CustomTF2/data/test.record'

# path to training folder
model_dir = '/content/gdrive/MyDrive/2CustomTF2/training'

# path to labelmap.pbtxt file
labelmap_path = '/content/gdrive/MyDrive/2CustomTF2/data/label_map.pbtxt'

# path to model config file
pipeline_config_path = '/content/gdrive/MyDrive/2CustomTF2/mobilenet_v2.config'

# path to ckpt-1 (within mobilenet folder, only include file path up til ckpt-1 or you will encounter errors)
fine_tune_checkpoint = '/content/gdrive/MyDrive/2CustomTF2/mobilenet_v2/mobilenet_v2.ckpt-1'

```


```python
# editing the config file - nothing needs changed here
import re

with open(pipeline_config_path) as f:
    config = f.read()

with open(pipeline_config_path, 'w') as f:

  # Set labelmap path
  config = re.sub('label_map_path: ".*?"', 
             'label_map_path: "{}"'.format(labelmap_path), config)
  
  # Set fine_tune_checkpoint path
  config = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
  
  # Set train tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                  'input_path: "{}"'.format(train_record_path), config)
  
  # Set test tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                  'input_path: "{}"'.format(test_record_path), config)
  
  # Set number of classes.
  config = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(num_classes), config)
  
  # Set batch size
  config = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), config)
  
  # Set training steps
  config = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(num_steps), config)
  
  f.write(config)
```


```python
# Load Tensorboard to analyze training 
%load_ext tensorboard

# path to training folder
%tensorboard --logdir '/content/gdrive/MyDrive/2CustomTF2/training'

```


```python
# On google colab atleast, opencv caused training issues,
# if you receive a cv2.cv2 error when training, uninstall and reinstall opencv
!pip uninstall opencv-python
```


```python
# Finally, to train the model. Run this code to begin training
# ensure in the first line that it lines up to the model_main_tf2.py file in your setup

!python /content/gdrive/MyDrive/2CustomTF2/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}
```


```python
# Once complete, run this code to evaluate the model

!python /content/gdrive/MyDrive/1CustomTF2/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --checkpoint_dir={model_dir}  
```
