# U-Net Brain Tumor Segmentation 

For the class of [Intelligent Systems for Pattern Recognition](https://elearning.di.unipi.it/course/view.php?id=110) a [U-Net Convolutional Network](https://arxiv.org/pdf/1505.04597.pdf) implemented in Pytorch was used to perform brain tumor segmentation. The model was trained and tested on the Kaggle platform in order to exploit GPU computation.  

The dataset was obtained from the [Cancer Image Archive](https://www.cancerimagingarchive.net/), a service that de-identifies and hosts a large archive of medical images of cancer accessible for public download. 
An analysis was done on the performance achieved by varying the hyperparameters of the model such as the batch size and regarding then use of the Batch Normalization layers. Also, a description of the model architecture used and the theoretical background of CNNs can be found inside the [report](https://github.com/FraCorti/U_Net-Brain-tumor-segmentation/blob/main/report.pdf).
