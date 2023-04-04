# Letter Recognition in Archaeology Writing using Deep Learning

This project aims to solve the problem of letter recognition in Archaeology writing using deep learning techniques. We will be using a self-supervised approach to achieve multi-class classification.

## Dataset Creation
We have created a DatabaseFactory class that will create a dataset from unorganized data folders. The data folders contain images of Archaeology writing in a structure of areas where they were found. Each image folder contains a PNG image along with a JSON file that contains relevant polygon annotation data for letters inside the image.


## Image Database Factory
Our code will crop and save the letter images in a sample folder in each image folder. It will also create a CSV file with relevant information about the cropped image and add relevant information on the original image to a separate CSV file.

## Requirements
Python 3
OpenCV
Pandas

## Usage
To use the script, simply instantiate a DatabaseFactory object with the path to the folder containing the images and annotations. 

## Example:
python
Copy code
from database_factory import DatabaseFactory
factory = DatabaseFactory("path/to/folder")

The script will automatically parse the JSON annotation files and extract information about the image samples. It will then save the information to two CSV files: 
samples_database.csv and original_images_database.csv.

## Functionality
The DatabaseFactory class has the following methods:

__init__(self, data_path)
The constructor method for the DatabaseFactory class. It takes a single argument, data_path, which is the path to the folder containing the images and annotations.

save_samples_data_to_csv(self)
This method saves the sample data and original image data to two separate CSV files.

parse_json_file(self, dirpath, filenames)
This method is responsible for parsing the JSON annotation files and extracting information about the image samples. It takes two arguments: dirpath, which is the path to the folder containing the annotation file, and filenames, which is a list of filenames in the folder.

parser_image_file(self, filenames)
This method extracts information about the original image file. It takes a single argument, filenames, which is a list of filenames in the folder.

fill_new_sample_info(self)
This method fills in the information for a new image sample.

parser_image_info(self, i)
This method extracts information about the image sample, such as the coordinates of the polygon.

annotate_image(self)
This method is used to annotate the original image with the polygon.

rectangle_crop_and_show(self)
This method is used to crop the image to the bounds of the polygon and display the cropped image.

get_polygon_crop(self)
This method returns a cropped version of the image based on the coordinates of the polygon.

