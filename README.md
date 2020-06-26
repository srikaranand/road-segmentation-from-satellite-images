## Files :
This code consists of 3 Python files:
`keras_pipeline.ipynb` - contains neural network used to make the model
`helpers.py` - consists of all the defined functions that we shall use to execute the program
`run.py` - generates the output from test images with the functions defined in helpers with weights given by keras_pipeline.ipynb
`csv2img.py` - generate the final output image(s) from the generated output csv file in run.py

## Dependencies :
TensorFlow,
Keras,
OpenCV,
imutils.


## Running :
## On Personal Computer :
The user simply needs to execute these commands in order in the Project Directory:
1. `python run.py`
2. `python csv2img.py`
## On Google Colab :
1. Mount the google drive to colab.
2. Upload the files of the dataset at this link to folder 'road-segmentation-dataset' in your Drive
    `www.kaggle.com/srikaranand/road-segmentation-dataset`
3. To run the program,  run cells #11 , #12 and #13  (Cell numbers at top of each cell)

The model was trained on Google Colab and the produced weights are saved to savedModels directory. The user can simply use these weights to directly run the testing file.

## Authors
Ishan Arora (IIT2017501)
Vikrant Singh (IIT2017502)
Srikar Anand (IIT2017504)
Akshay Gupta (IIT2017505)
Naman Deept (IIT2017507)