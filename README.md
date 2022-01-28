# P10-Code
Remember to update paths.csv to fit with local paths to files/folders.
All files are python files, and package requirements are listed in requireents.txt
The project also has a dockerfile to build a container.

'article_extraction.py' and 'traverse.py' are used to load the data from many raw .xml files to a single .json file, with only extracted information.

'preprocessing.py' is used to clean up the information of the .json file and generate files used by the models

Finally, the different .py files in the 'model' folder are used to run the actual topic models.
The primary used model is 'pachinko_gibbs_lda.py'