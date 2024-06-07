# Reproducing Results for Naive Bayes, BERT, and DistilBERT Training on Amazon, Movie, and Restaurant Reviews 
### Authors: Ayman Khan, Mohamad Abbas, and Prashanth Babu

This file contains information about the code and notebooks to reproduce the main results in our paper. The experiments involve training BERT, Naive Bayes, and DistilBERT models on Amazon, movie, and restaurant reviews datasets. 

## Prerequisites

Before running the codes, ensure you have the following installed:
- Python 3.7 or higher
- PyTorch
- Transformers library from HuggingFace
- Pandas
- Scikit-learn
- Jupyter Notebook

For running or testing the BERT and DistilBERT models you need to be using some sort of GPU such as ada clustering, where you can submit the training file as a slurm script, then will give you output files with your results.

## Dataset

Ensure you have the Amazon, movie, and restaurant reviews datasets downloaded and placed in the folder. You can modify the paths in the scripts if your data is located elsewhere.

## Training BERT model on all the datasets

Make sure to import the BERT model from Bert.py into the training files.

The code files berttrain_amazon.py, berttrain_movie.py, and berttrain_rest.py are the files where all the training and testing is happening. The scripts preprocesses the data, trains a BERT model, and saved the trained model. To run the scripts we open the train.sbatch file and change the filename to the corresponding bert file we want to train. Then we run the following command after saving the changes: sbatch train.sbatch. An output file will be displayed with all the epochs, accuracies, and learning rates. The results of the BERT models that we ran are in the BertResults folder. 

## Training DistilBERT model on all the datasets

Make sure to import the DistilBERT model from DistillBert.py into the training files.

The code files distilberttrain_amazon.py, distilberttrain_movie.py, and distilberttrain_rest.py are the files where all the training and testing is happening. The scripts preprocesses the data, trains a DistilBERT model, and saved the trained model. To run the scripts we open the train.sbatch file and change the filename to the corresponding distilbert file we want to train. Then we run the following command after saving the changes: sbatch train.sbatch. An output file will be displayed with all the epochs, accuracies, and learning rates. The results of the DistilBERT models that we ran are in the DistilBertResults folder. 

## Training NaivesBayes Model on all the datasets

Make sure to import the NaiveBayes model from NaiveBayes.py into the training files.

The code files amazontrainedcodenb.ipynb, movietrainedcodenb.ipynb, and restauranttrainedcodenb.ipynb are the files where the detailed analysis and step-by-step execution of the Naive Bayes model happens. The cleaning of the data and mutating new columns happens in these files, where we train and store the models also. To run the file you need to run the code block by block, and you will see the step by step outputs of the codes.

## Acknowledgements

We would like to thank the HuggingFace team for providing the Transformers library and the authors of the datasets used in this research.

## Citations

Movie review Dataset: Jillanisofttech. “IMDB Movie Reviews 50K.” Kaggle, Kaggle, 30 June 2022, www.kaggle.com/code/jillanisofttech/imdb-movie-reviews-50k/input. 

Amazon review Dataset: J, Karkavelraja. “Amazon Sales Dataset.” Kaggle, 17 Jan. 2023, www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset?resource=download. 

Restaurant review Dataset: Anwar, Arsh. “Restaurant Reviews.” Kaggle, 7 June 2021, www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews. 

