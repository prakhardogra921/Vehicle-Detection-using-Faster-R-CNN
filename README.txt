Author: Prakhar Dogra

The following README gives details about the dataset and the files contained in this folder:

1. Dataset
	Dataset can be downloaded from the following weblink:
	http://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar
	
	Moreover, the pretrained architecture can be downloaded from:
	VGG: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
	ResNet: https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
	
2. SourceCode
	This folder contains the Source Code of the implementation for this project:
	- train_frcnn_resnet.py: Implementation of training procedure when using ResNet architecture.
	- test_frcnn_resnet.py: Implementation of testing procedure when using ResNet architecture.
	- train_frcnn_vgg.py: Implementation of training procedure when using VGG architecture.
	- train_frcnn_vgg.py: Implementation of testing procedure when using VGG architecture.
	- map.py: Procedure that calculates the mean average precision score.
	- faster_rcnn:
		- data_augment: Implementation for data augmentation
		- data_generators: Functions for using ground truth bounding boxes
		- fixed_batch_normalisation.py: Functions and classes for batch normalization
		- intersection_over_union.py: Functions for calculating IoU values
		- losses.py: Functions for calculating the bounding box regression and classification losses
		- parser.py: Implementation for parsing the image files and ground truth labels
		- resnet.py: Functions for creating and using the ResNet architecture
		- roi_helpers.py: Helper functions for ROI pooling
		- roi_pooling_conv.py: Functions implementing ROI pooling.
		- vgg.py: Functions for creating and using the VGG architecture
		- visualize: Contains function that helps draw bounding boxes on the images

3. Output
	This folder contains the two sub-folders of each respective method implemented:
	a. sample_resnet_results:  Contains 20 sample images that were obtained when ResNet architecture was used.
	b. sample_vgg_results:  Contains 20 sample images that were obtained when VGG architecture was used.

4. Report and PPT
	This folder contains the Report and Presentation of the Project.
	
5. Compiling Instructions
	After downloading and unzipping the dataset make sure to place the dataset inside the src directory. Also create empty folders "model" to store the pretrained model weights and "model_trained" to save the weights of training procedure.

6. References
	- https://github.com/yhenon/keras-frcnn
	- https://github.com/keras-team/keras/tree/master/keras/applications
	- https://github.com/jinfagang/keras_frcnn
