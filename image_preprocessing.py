import numpy 
import joblib
import matplotlib.image as img
import glob 
from sklearn.model_selection import train_test_split

samples = []
targets = []

images_directories = ["mask_weared_incorrect", "with_mask", "without_mask"]

target_id = 0
for im_dir in images_directories:
    for im in glob.glob(im_dir + "/*.png"):
        print("working in {}".format(im))
        image = img.imread(im)
        samples.append(image)
        targets.append(target_id)
    target_id += 1

samples = numpy.hstack([samples])
targets = numpy.array(targets)

train_data, test_data, train_labels, test_labels = train_test_split(samples, 
                                            targets, 
                                            test_size = 0.20, 
                                            stratify=targets)

joblib.dump([train_data, test_data, train_labels, test_labels], "facemask_dataset.pkl")
