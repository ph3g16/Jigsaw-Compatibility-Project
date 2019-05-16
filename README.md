Use this code at your own risk. Permission is given for non-commercial use only, no warranty is provided, and no liability will be accepted for damage caused by use or misuse.

# Jigsaw-Compatibility-Project

Code used to generate results for MATH3031 project: Jigsaws and Application of Neural Networks to Edge Matching

The code is a bit of a mess and needs some tidying up. Notably, a functional menu is not yet complete. The repository will be cleaned up after marks are awarded for the university project.

Currently functions are spread across four files:

<b>Image_Prep.py</b> - this contains most of the functions used to process images and generate training/test data.

<b>Matching_NN.py</b> - the most coherent and best commented file. This contains the neural network definition and associated functions such as training, evaluation, and checkpoint saves.

<b>Menu.py</b> - this contains routines for performing cross-validation plus a non-functional menu.

<b>Norm_Calc.py</b> - an unfinished module with functions for calculating the various norms used as compatibility measures in other papers.
