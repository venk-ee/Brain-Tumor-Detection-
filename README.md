Brain Tumor Detection using VGG-16 CNN Model

Overview
This project aims to develop a Convolutional Neural Network (CNN) model for classifying whether a subject has a brain tumor or not based on MRI scans. The model utilizes the VGG-16 architecture and pre-trained weights for training on a binary classification task. Additionally, a web user interface (UI) is provided to facilitate user inputs for tumor detection.
Model Performance

The model's performance is evaluated using accuracy as the metric:

    Validation Set Accuracy: Approximately 88%
    Test Set Accuracy: Approximately 80%

Usage

Dependencies

Ensure you have the following dependencies installed:

    Python 3.x
    TensorFlow
    Keras
    Flask (for the web UI)

Training the Model

    Run mainTrain.py script to train the CNN model using the VGG-16 architecture and pre-trained weights.
    The trained model will be saved as BrainTumor10Epochs.h5.

Testing the Model

    Run mainTest.py script to test the trained model on test data.
    Evaluate the model's performance on the test set.

Web UI

    Ensure you have Flask installed (pip install flask).
    Run the app.py script to start the Flask web server.
    Access the web UI through your browser and provide the required inputs for tumor detection.

Further Improvements

    Fine-tuning Model Parameters: Experiment with different hyperparameters to potentially improve model performance.
    Data Augmentation: Augment the training data to increase diversity and improve model generalization.
    Ensemble Methods: Explore ensemble techniques to combine predictions from multiple models for enhanced accuracy.
    Clinical Validation: Conduct rigorous validation studies to evaluate the model's performance on real-world clinical data.
    User Interface Enhancements: Improve the user interface for better usability and accessibility.

Contributors

    VENKATANATHA AV [CNN]
    NINAD AITHAL[WEB UI]
    

