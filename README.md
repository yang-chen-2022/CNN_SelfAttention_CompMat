# CNN_SelfAttention_CompMat
Apply a Convolutional Neural Network with self-attention (CNNSA) to predicting full field damage in composites

  - Implementation based on PyTorch
  - Required packages: 
    - _imgaug_ for data augmentation
    - _vtk_ for importing image data
  - Pre-trained models provided

You can test the performance of the pre-trained model (saved in _models_). Do do so, you should download the four directories: _code_, _data_, _models_, _runs_

If you wish to re-train the model for your own data, you need either to arrange the data structure to be consistent with what I used (see _code/utils/IOfcts.py_ for details), or to write a new data loader that matches your data structure. I would suggest the latter, as it would be easier ;-)

More details about the CNN_SA model can be found in our paper:

Chen, Y., Dodwell, T., Chuaqui, T., & Butler, R. (2023). Full-field prediction of stress and fracture patterns in composites using deep learning and self-attention. Engineering Fracture Mechanics, 286, 109314. https://doi.org/10.1016/j.engfracmech.2023.109314
