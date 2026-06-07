#![allow(dead_code)]

/*

    NeuralNet library for building and training neural networks, including CNNs and NODENets.

    This library provides core components for defining neural network architectures, training loops, and utilities for handling multimodal data.
    The library is organized into several modules:

    - `core`:       Contains the core components of the neural network, including the main brain structure, model definitions, and training logic.
    - `cnn`:        Contains components specific to convolutional neural networks (CNNs), including feature extractors, classifiers, and CNN trainers.
    - `dendrites`:  Contains definitions for different types of dendrites used in the neural network.
    - `helpers`:    Contains helper functions and utilities for working with multimodal data and other common tasks.
    - `rag`:        Contains retrieval-augmented generation interfaces and orchestration that compose with the core brain.
    - `tensor`:     Contains definitions and utilities for working with tensors, including 4D tensors for CNNs and image processing utilities.
    - `training`:   Contains the training loop implementation and related utilities for training neural networks.

    This is supplied under the GPL-3.0 license. See LICENSE file in the project root for full license text.
    Author: Sam Colak (sam@samcolak.com)

*/

pub mod core;
pub mod cnn;
pub mod dendrites;
pub mod helpers;
pub mod rag;
pub mod tensor;
pub mod training;
