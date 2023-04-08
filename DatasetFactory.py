import cv2
import json
import numpy as np
import os
import pandas as pd


class DatasetFactory:
    def __init__(self, data_path):

        self.samples_data = {}
        self.original_images_data = {}
        self.samples_amount = 0

        for dirpath, dirnames, filesnames in os.walk(data_path):  # search in database folder
            if filesnames:  # means there is photos in the folder
                self.image_info = {}
                self.parser_image_file(filesnames)
                self.parse_json_file(dirpath, filesnames)
        self.save_samples_data_to_csv()

    def save_samples_data_to_csv(self):
        df1 = pd.DataFrame.from_dict(self.samples_data, orient='index')
        df1.to_csv('samples_database.csv')

        df2 = pd.DataFrame.from_dict(self.original_images_data, orient='index')
        df2.to_csv('original_images_database.csv')

    def parse_json_file(self, dirpath, filesnames):
        for filename in filesnames:
            if filename.endswith("json"):
                print(f"this is image number: {self.samples_amount}, file name: {filename}\n")
                self.image_info["original_folder_path"] = dirpath
                self.image_info["json_name"] = "\\" + filename
                with open(self.image_info["original_folder_path"] + self.image_info["json_name"], 'r') as f:
                    self.data = json.load(f)
                    f.close()
                for i in range((len(self.data['annotations'][0]['result']))):
                    self.parser_image_info(i)

                    cropped = self.get_polygon_crop()
                    if not os.path.exists(self.image_info["original_folder_path"] + "\\Samples"):
                        os.makedirs(self.image_info["original_folder_path"] + "\\Samples")

                    self.new_sample_info = {}
                    self.fill_new_sample_info()
                    if self.image_info:
                        self.samples_data[self.samples_amount] = self.new_sample_info
                        self.original_images_data[self.image_info["original_folder_path"]] = self.image_info
                    self.samples_amount += 1
                    cv2.imwrite(self.new_sample_info["sample_path"], cropped)

    def parser_image_file(self, filesnames):
        for filename in filesnames:
            if filename.endswith("png"):
                self.image_info["original_image_name"] = "\\" + filename

    def fill_new_sample_info(self):
        self.new_sample_info["sample_name"] = str(self.samples_amount)
        self.new_sample_info["from_image"] = self.image_info["original_image_name"][1:]
        self.new_sample_info["from_area"] = self.image_info["original_folder_path"].rsplit("\\", 3)[-2]
        self.new_sample_info["sample_path"] = self.image_info["original_folder_path"] + "\\Samples\\" \
                                              + self.new_sample_info["sample_name"] + '.png'
        self.new_sample_info["label"] = self.new_sample_info["from_area"] + "_-_" + self.new_sample_info[
            "from_image"]

    def parser_image_info(self, i):
        self.image_info['points'] = self.data['annotations'][0]['result'][i]['value']['points']
        self.image_info['width'] = self.data['annotations'][0]['result'][i]["original_width"]
        self.image_info['height'] = self.data['annotations'][0]['result'][i]["original_height"]
        self.image_info['scaled_points'] = np.array(
            [[int(x / 100.0 * self.image_info['width']), int(y / 100.0 * self.image_info['height'])]
             for x, y in self.image_info['points']], dtype=np.int32)

    def annotate_image(self):
        pass
        # scaled_points = self.get_scaled_points()
        # cv2.polylines(self.image, [scaled_points], True, (0, 255, 0), thickness=20)
        # cv2.imshow('image', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def rectangle_crop_and_show(self):
        pass
        # x, y, w, h = cv2.boundingRect(np.array([self.scaled_points], dtype=np.int32))
        # cropped = self.image[y:y + h, x:x + w]
        # cv2.imshow('Cropped Image', cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def get_polygon_crop(self):
        image = cv2.imread(self.image_info["original_folder_path"] + self.image_info["original_image_name"])
        x, y, w, h = cv2.boundingRect(np.array([self.image_info['scaled_points']], dtype=np.int32))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [self.image_info['scaled_points']], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cropped = masked_image[y:y + h, x:x + w]
        cropped[mask[y:y + h, x:x + w] == 0] = 255
        return cropped


DatasetFactory((r"C:\Users\yarin\PycharmProjects\GuidedProject\Database"))
