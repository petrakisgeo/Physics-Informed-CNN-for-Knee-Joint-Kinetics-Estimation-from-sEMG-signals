# Physics Informed CNN for Knee Joint Kinetics Estimation from sEMG signals
 CNN surrogate model for estimation of biomechanic parameters of the knee from 4-11 sEMG sensor signals in multiple walking conditions. We use the OpenSim Python API Inverse Dynamics tool to add a physics loss parameter to the network, using batch processing. This repository can also serve as a framework for handling the Epic Lab Dataset (https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) with Python instead of Matlab.

Segmentation to gait cycles and sEMG signal preprocessing is implemented in the "gatherData.py".
CNN architecture can be seen below. The feedforward layer architecture was determined by grid-search with the Optuna Python package, as 1 hidden layer with 512 neurons.
<p align="center">
<img src="https://github.com/petrakisgeo/Physics-Informed-CNN-for-Knee-Joint-Kinetics-Estimation-from-sEMG-signals/assets/117226445/e9034fee-7196-4e49-8705-b9cdf464cdea">
</img>


The above network, without physics information, is tested on 20 subjects in Leave-Trials-Out and Leave-Subject-Out cross-validation scenarios. Also there are tests showcasing 

The framework for the physics informed loss calculation can be seen below. For each training batch, the predicted knee angles are stored in a .sto file and then read by the OpenSim ID tool along with other data and subject-specific musculoskeletal models to calculate the knee joint moment. The MSE between the physics-calculated loss and the neural network one is the physics loss of the network.
<p align="center">
<img scr="(https://github.com/petrakisgeo/Physics-Informed-CNN-for-Knee-Joint-Kinetics-Estimation-from-sEMG-signals/assets/117226445/1d5f932f-ace5-4934-8daa-01ac8f487df0)">
</p>
We test the assumption that physics information helps a neural network converge faster in low training data scenarios. The test is performed on 20% of the data in a Leave Trials Out scenario on 20 subjects
Our results indicate the following:
 1. Physics information does not show defining results for improvement of training convergence. Bias during training suggests further research is required on adaptive PINNs
 2. The OpenSim batch processing framework is not fit for this type of applications. The repeated read/write operations slow the training time down dramatically and could be easily avoided with slight API adjustments. There is also a case made for memory leakage during InverseDynamicsTool.run() main loop. Even after deleting the Tool object manually, some amount of memory still persists, causing a memory leak on repeated calls of the run() method.
