# neural-network-and-deep-learning-with-MATLAB
I'm a start learner of deep learning and I found Michael Nielsen's online e-book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) awesome! In the meantime I'm a **_MATLABer_** with 10 years experiences. In this project, I'm gonna rewrite what Michael has done in Python using MATLAB. I'm doing so partly to digest Michael's book and partly for other MATLAB users to read and enjoy this book.

**File content**

  **nnet.m**: _corresponding to network.py_
  
  **nneto.m**: _a further vectorized version of nnet.m, not corresponding to any code in Michael's book. But Michael did commented something related to further vectorization of network.py_
  
  **test_nnet_MNIST.mlx**: _this is MATLAB live script that includes runing output. Using the same configuration as Michael illustrated in his book. i.e., net size [784, 30, 10] This takes around 600 sec to finish 30 epochs_
  
  **__NNET_MNIST_README_20190118.txt**: _a readable version for results shown in **test_nnet_MNIST.mlx**. Note they were from different trials so not exactly the same_
  
  **test_nneto_MNIST.mlx**: _live script again but using net size [784, 100, 10] This only takes less than 100 sec with the additional vectorization to finish 30 epochs_
  
  **__NNETO_MNIST_README_20190118.txt**: _systemwide readable results based on net size [784, 30, 10] but using **nneto.m** for comparing the speed of training by further vectorization with **nnet.m** (shown in **test_nnet_MNIST.mlx** or **__NNET_MNIST_README_20190118.txt**)_

  **gnnet.m**: _corresponding to network2.py (updated from **nneto.m**). It requires two additional class definitions **CrossEntropyCost.m** and **QuadraticCost.m**_

  **CrossEntropyCost.m**: _class definition for cross entropy cost function_

  **QuadraticCost.m**: _class definition for quadratic cost function_

  **folder: Weight_init_regularization_test**: _test resulting using large_weight_init with or without regularization, and using small_weight_init with regularization_

  **test_gnnet_MNIST.m**: _this the program I used to test the **gnnet.m**. It will give you a log ASCII file showing the steps of training ect and an MATLAB figure output showing the cost functions and accurracies. I used this to produce the results in **folder: Weight_init_regularization_test** with some change in parameters in this program_

  **test_toolbox_shallow.html**: _**this file is within test_toolbox.zip** program and results for using MATLAB Deep Learning Toolbox to reconstruct a shallow neural network model as what we've built from scratch_

  **prepareData2Local.m**: _used to save MNIST data to local drive_

  **sigmoidLayer.m**: _you may need this when you want you implement a sigmoid layer in deep neural network architecture_

  **crossEntropyClassificationLayer.m**: _this is added in order to use a sigmoidLayer for output activation. It doesn't work as I explained in the **test_toolbox_shallow.html** at this moment since MATLAB keeps complaining about output activation_

  **test_toolbox_deep_1.html**: _**this file is within test_toolbox.zip** this model corresponds to the first deep neural network Michael introduced (with 1 convolutional layer). Test accuracy reached to 98.2% in my first try_

  **test_toolbox_deep_2.html**: _**this file is within test_toolbox.zip** this model corresponds to the model Michael introduced with 2 convolutional layers. Test accuracy reached to 99% in my first try. I've also implemented another technique called batch normalization in this model construction. See this file for a bit more info_

  **test_toolbox_deep_augmentation**: _results based on the same model as we've used in **test_toolbox_deep_2.html**. However, I've done some image augmentation before training. It give a 99.4% accuracy in my first try_