# Voice Access Control

This repository contains a computer vision project for voice-based access control. The system receives a speech recording and decides whether the speaker belongs to the allowed group.

The audio recordings are preprocessed, split into short segments, and converted into PCEN spectrograms. These spectrograms are treated as image-like inputs for convolutional neural networks. Several CNN families are compared, including a plain CNN, residual CNN, lightweight mobile-style CNN, and transfer-learning-based CNN.

The final system uses a ResCNN model in prototype mode. In this mode, the CNN acts as an embedding extractor, and each allowed speaker is represented by a stored prototype vector. A new speaker can be added by recording or uploading several samples and updating the prototype index, without retraining the whole CNN.

The repository includes:

- data preprocessing and spectrogram generation pipeline,
- CNN training and model comparison notebooks,
- hyperparameter tuning with Optuna,
- final model evaluation,
- noise robustness experiments,
- Streamlit application for access checking and speaker enrollment.
