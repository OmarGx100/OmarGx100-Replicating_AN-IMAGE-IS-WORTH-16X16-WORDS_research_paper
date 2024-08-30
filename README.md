
---

# Vision Transformer (ViT) Implementation

This repository contains the implementation of the Vision Transformer (ViT) model, inspired by the research paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929). The code replicates the ViT architecture, which applies the Transformer model—originally designed for natural language processing—to image classification tasks.

## Project Overview

### Dataset
The model was trained on a small dataset consisting of only 600 images, divided into three classes. This limited dataset size significantly impacts the model's performance. The original Vision Transformer model was trained on large-scale datasets, such as ImageNet, with millions of images, allowing it to achieve state-of-the-art accuracy.

### Model Architecture
The ViT model is composed of several key components:
1. **Patch Embedding**: The input images are divided into 16x16 patches, which are then flattened and linearly transformed into embedding vectors.
2. **Positional Embeddings**: Since the Transformer architecture doesn't inherently capture the spatial structure of the image, positional embeddings are added to the patch embeddings to retain positional information.
3. **Transformer Encoder**: The core of the ViT model, consisting of multiple Transformer Encoder layers, each with a Multi-Head Self-Attention mechanism and a Multi-Layer Perceptron (MLP) block.
4. **Classification Head**: The output of the Transformer is passed through a fully connected layer to produce the final class predictions.

### Training and Evaluation
The model was trained and evaluated using standard PyTorch training loops. Due to the small dataset size, the model's accuracy is lower than expected. The original ViT model benefits from extensive training on large datasets, which is a limitation of this implementation.

### Results
- **Training Accuracy**: The training accuracy improved with each epoch but was limited by the small dataset.
- **Testing Accuracy**: Testing accuracy reflects the model's generalization capabilities, which were constrained by the limited training data.


## Future Work
- **Dataset Expansion**: Improving model accuracy would require training on a larger dataset.
- **Model Tuning**: Hyperparameter tuning and increasing the number of Transformer layers could improve performance.

## Acknowledgements
This implementation is heavily based on the paper "An Image is Worth 16x16 Words," which introduced the Vision Transformer. The original ViT model was trained on the JFT-300M dataset containing 300 million images, achieving remarkable performance across various benchmarks.

## License
This project is licensed under the MIT License.

---
