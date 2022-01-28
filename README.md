# P10-Code
https://github.com/MrLongbottom/P10-Code/tree/archiving

Original test environment was on windows using PyCharm to setup virtual environment. Has also been run on a linux server using docker with the Dockerfile provided within.

How to run:
Use a python interpreter to run .py files. Package requirements are listed in 'requirements.txt'.
The project also has a dockerfile to build a container.

Remember to update paths.csv to fit with local paths to files/folders.

the dataset used is 'Nordjyske' news articles from 2017-2019 kept in .xml format.

'preprocess' folder is to get the data ready to run model, it has its own 'generated_files' folder with data ready to be used by the models.

'model' folder is to run topic models. Once done, resulting model is saved in the 'generated_files' sub-folder.

'exploration' folder consists of minor experiments and visualizations used in the original 10th semester project report. They are not particularly useful, but are kept here in case they are informative about the data processing.

'evaluation' folder has a single 'prediction.py' file, used to try to predict author based on topic distributions of a document. However this has proved to be largely unsuccessful.

'article_extraction.py' and 'traverse.py' are used to load the data from many raw .xml files to a single .json file, with only extracted information.

'preprocessing.py' is used to clean up the information of the .json file and generate files used by the models

Finally, the different .py files in the 'model' folder are used to run the actual topic models.
The primary used model is 'pachinko_gibbs_lda.py'