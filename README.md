# Archaeological Inscription Recognition Project

## Project Overview

Creating good datasets for deep learning tasks is tough, especially when it takes a lot of time and money. In archaeology, it's even more difficult because you need experts who understand the ancient writing to label the data, which often has lots of imperfections. Our project aims to make this easier, which could be a big help for archaeology research.

We're trying to solve this by focusing on recognizing individual letters on artifacts using a self-supervised approach – specifically a technique developed by FAIR called DINO. This approach tries to identify and pull out the letters automatically.

In this repository, you'll find a notebook that shows our early attempts to solve this complex issue. We hope it will motivate and help others who are working on similar problems.

## Liberis
To run this notebook, you'll need to install several libraries. You can install these via pip or conda, depending on your setup. Here's a list of the required libraries:
- OpenCV (cv2): For image processing tasks.
- pandas: For handling data and saving it in CSV format.
- numpy: For numerical operations on arrays and matrices.
- os: For interacting with the file system.
- json: For parsing JSON files that contain annotations.
- pytorch_lightning: For organizing the deep learning workflow, making it more structured and providing high-level components for training.
- torch: PyTorch, for building and training neural networks.
- torchvision: Provides access to datasets, models, and image transformations for computer vision.
- lightly: A computer vision framework for self-supervised learning.

## Done Tasks
- **Annotation with Label-Studio**: Annotated 400 images with polygonal labels, along with JSON metadata for future reuse.
- **Image Cropping**: Developed Python code to crop images based on annotations.
- **Custom PyTorch Dataset**: Created a PyTorch dataset class for handling annotated data.
- **DINO Model with Lightly.ai**: Built a DINO model and studied SSL techniques, including SimCLR and BYOL.
- **Data Transformations**: Engineered a data transformation pipeline for the DINO model.
- **Pretraining Tasks**: Pretrained the model with a focus on learning rates and overfitting a single batch.
- **Training on Google Colab**: Trained the model for 180 epochs using Colab's GPU resources.
- **Initial Results with KNN**: Evaluated initial results using K-nearest neighbors on training dataset embeddings.

## Future Tasks
- Train the model for downstream tasks like detection or segmentation.

## Augmentation Strategies
- **Camera Position Adjustments**: Implemented `torchvision.transforms.RandomRotation` for simulating changes in camera angle through rotation.
- **Crack Simulation**: Explored options for creating cracks but deferred implementing a custom transformation for future work.
- **Vintage Photo Transformation**: Applied `torchvision.transforms.ColorJitter` to adjust color balance for an "aged" photo effect.
- **Arbitrary Deletions**: Utilized `torchvision.transforms.RandomResizedCrop` to imitate random crops and deletions within images.
- **Lighting Variability**: Employed `torchvision.transforms.ColorJitter` to emulate variations in image brightness, akin to changing lighting conditions.
## How to Use
Detailed in ArchaeologyLettersRecognition.ipynb notebook.

## Contributors
Special thanks to Dr. Barak Sober from The Hebrew University of Jerusalem's Department of Statistics and Data Science and Digital Humanities ([personal website](https://barakino.wixsite.com/academics)) and Prof. Yedid Hoshen from the School of Computer Science and Engineering at the same university ([personal website](https://www.cs.huji.ac.il/~yedid/)), for their invaluable guidance and support throughout this project.

## Acknowledgements
Our project is heavily inspired by the work of Mathilde Caron et al. on Self-Supervised Vision Transformers (https://arxiv.org/abs/2104.14294).
For more information on their work, please visit the DINO GitHub repository.
