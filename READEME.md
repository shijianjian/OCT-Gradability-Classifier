# OCT Gradability Classifier

Code repository for paper [Artificial intelligence deep learning algorithm for discriminating ungradable optical coherence tomography three-dimensional volumetric optic disc scans](./doc/paper.pdf).

## Requirements
- tensorflow >= 1.13.0
- OpenCV
- scikit-image >= 1.15.0

## Usage
- Put new data into the folder ```sample_data``` accordingly by the folder structure with a csv datasheet:
    ```
    - sample_data
        - dataset
            - gradable
            - ungradable
        - datasheets
            - train.csv
            - test.csv
            - predict.csv
    ```
- For training or evaluating, denoise all the data to avoid denoising on the fly:
    ```
    $ python denoising.py
    ```
- Training
    ```
    $ python index.py
    ```
- For testing or prediction using an existed model:
    1. Add a file named ```latest``` to the ```trained_model``` folder to indicate which model to use.
    2. Put ```predict.csv``` into ```./sample_data/datasheets/predict.csv```
    3. Run
        ```
        $ python index.py --predict --denoise --folder="./trained_model"
        ```
        Note: the uploaded trained model was trained using 4 gpus and was saved as a ```multi_gpu_model``` from tensorflow keras.