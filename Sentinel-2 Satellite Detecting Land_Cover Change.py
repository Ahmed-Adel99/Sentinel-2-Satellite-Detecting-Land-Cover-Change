import random  # I have total of 100 points, it takes 70 points for train samples and the rest for test samples.
from osgeo import gdal, ogr  # gdal: reads raster data(layer), ogr: reads vector layers(shape files).
import numpy as np  # The raster layer will be converted to numpy(np) array.
from sklearn.svm import SVC  # Imports object to train with the train samples and test with the test sample and print accuraccy and also import the support vector machine algorithm to be used.
from sklearn.tree import DecisionTreeClassifier    # Import the decision tree algorithm.
from sklearn.cluster import KMeans    # Import the k means algorithm.
from sklearn import metrics # It generates the confussion matrix and the accuraccy score.
from skimage.io import imsave # It takes the numpy array and generate an image and save it on the hard disk.
import json     # A way of communication between python and java, by table and report.


# Global variables to allow any of the other scripts to read them ex:run_svm.py.
# Five classes used to read the shape files from the hard disk (must be the same name of the shape files).
# classes_names = ["Veg", "Water", "Desert", "Bare_Soil", "Urban", "BG"]  
classes_names = ["vegtation", "water", "desert", "roads", "urban", "BG"]  
# classes_names = ["vegtation", "water", "desert", "roads", "urban"]  

# Classes of the k means algorithm.
classes_names_unsupervised = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10"]    

# The attributes that will be generated in the table of the report.
json_names = ["order_id", "process", "classifier", "number_of_classes", "names_of_classes", "data_dimensions", "confusion_matrix", "accuracy_score", "number_of_pixels_per_class", "date1_old_date", "date2_new_date", "number_of_pixels_per_state"]

# Classes for the supervised algorithms.
switcher = {
        1: (0, 255, 0),  # Veg
        2: (0, 0, 255),  # Water
        3: (255, 255, 0),  # Desert
        4: (160, 80, 40),  # Bare_Soil
        5: (127, 127, 127),  # Urban
        6: (0, 0, 0),  # BG(Background)
    }
# Classes for the unsupervised algorithm which is k means algo.
switcher_unsupervised = {
        1: (255, 0, 0),  # class 1
        2: (0, 255, 0),  # class 2
        3: (0, 0, 255),  # class 3
        4: (0, 255, 255),  # class 4
        5: (255, 0, 255),  # class 5
        6: (255, 255, 0),  # class 6
        7: (85, 255, 170),  # class 7
        8: (170, 85, 255),  # class 8
        9: (255, 170, 85),  # class 9
        10: (127, 127, 127),  # class 10
    }

# Classes for the change detection algorithm(6 classes for the first classified image another 6 classes for the second classified image so the total is 6x6=36).
switcher2 = {
        1: (128, 255, 128),  # Veg-Veg
        2: (255, 0, 255),  # Veg-Water
        3: (255, 0, 255),  # Veg-Desert
        4: (255, 160, 128),  # Veg-Bare_Soil (special case)
        5: (255, 0, 255),  # Veg-Urban
        6: (0, 0, 0),  # Veg-BG
        7: (255, 0, 255),  # Water-Veg
        8: (128, 128, 255),  # Water-Water
        9: (255, 0, 255),  # Water-Desert
        10: (255, 0, 255),  # Water-Bare_Soil
        11: (255, 0, 255),  # Water-Urban
        12: (0, 0, 0),  # Water-BG
        13: (255, 0, 255),  # Desert-Veg
        14: (255, 0, 255),  # Desert-Water
        15: (255, 255, 128),  # Desert-Desert
        16: (255, 0, 255),  # Desert-Bare_Soil
        17: (255, 0, 255),  # Desert-Urban
        18: (0, 0, 0),  # Desert-BG
        19: (128, 255, 128),  # Bare_Soil-Veg (special case)
        20: (255, 0, 255),  # Bare_Soil-Water
        21: (255, 0, 255),  # Bare_Soil-Desert
        22: (255, 160, 128),  # Bare_Soil-Bare_Soil
        23: (255, 0, 255),  # Bare_Soil-Urban
        24: (0, 0, 0),  # Bare_Soil-BG
        25: (255, 0, 255),  # Urban-Veg
        26: (255, 0, 255),  # Urban-Water
        27: (255, 0, 255),  # Urban-Desert
        28: (255, 0, 255),  # Urban-Bare_Soil
        29: (200, 200, 200),  # Urban-Urban
        30: (0, 0, 0),  # Urban-BG
        31: (0, 0, 0),  # BG-Veg
        32: (0, 0, 0),  # BG-Water
        33: (0, 0, 0),  # BG-Desert
        34: (0, 0, 0),  # BG-Bare_Soil
        35: (0, 0, 0),  # BG-Urban
        36: (0, 0, 0),  # BG-BG
    }
    
# Calling this function in all the processes down below, It reads the image band by band and stores it in an numpy array and putting it in this function which is called open_dataset and then with the gdal i open the image.
def open_dataset(data_file_path):    
    # Opening dataset.
    ds = gdal.Open(data_file_path)    # The image path.
    geo_transform = ds.GetGeoTransform()
    proj = ds.GetProjectionRef()
    bands_data = []    # Create an empty bands_data and fill it band by band.
    print('Reading Bands Data')
    for band in range(ds.RasterCount):    # Number of bands in the image.
        band += 1    # gdal starts reading from 1 to 7.
        nparray = ds.GetRasterBand(band).ReadAsArray()    # It takes the band that has the turn and read it as numpy array.
        print('\r\tReading Band #', band)
        bands_data.append(nparray)    # Putting the nparray with the empty bands data and will continue to loop untill it reach to 7.

    ds = None  # Release the memory to free space.
    bands_data = np.dstack(bands_data)  # Convert list of layers into 3d matrix, convert from list objects to numpy array, putting it into the same variable as not to take space in the RAM.
    return bands_data, geo_transform, proj


def extract_shapefiles_pixels(dir_shp, bands_data, geo_transform, proj, cols, rows):
    # Labeled_train_pixels: pixels of train set.
    # Labeled_test_pixels: pixels of test set.
    labeled_tr_pixels = np.zeros((rows, cols))  # create numpy array of train pixles to the layer that is formed at the max upper layer of the image and put it with zeros.
    labeled_ts_pixels = np.zeros((rows, cols))    # create numpy array of test pixles to the layer that is formed at the max upper layer of the image and put it with zeros.

    file_order = 1  # Index for each class start at 1 represents Veg.shp.
    perc_train_ratio = 70  # Ratio here is 70% of class pixels is train and the remaining 30% is to test.
    print('\t2. Reading Classes')
    
    for class_name in classes_names:
        print('\tClass #' + str(file_order) + ': ' + class_name)
        shp_file = dir_shp + '/' + class_name + '.shp'    # File name of the current class.
        file = ogr.Open(shp_file) # Open the shapefile(ogr opens vector data).
        
        if file is None:
            print('\t\tCould not open %s' % (shp_file))
        else:
            print('\t\tOpened %s' % ('shape file: ' + class_name + '.shp'))

        data_source = gdal.OpenEx(shp_file, gdal.OF_VECTOR)  # open vector using gdal(raster)
        
        if data_source is None:
            print("\t\tFile read failed: %s", shp_file)
        else:
            print("\t\tFile %s read successfully. " % (class_name + '.shp'))

        layer = data_source.GetLayer(0)
        driver = gdal.GetDriverByName('MEM')  # Driver of shape file, create a memory raster.
        target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)  # Create new dataset with the same dimensions of the image.
        target_ds.SetGeoTransform(geo_transform)   # It takes the transformation and put it onto the new dataset.
        target_ds.SetProjection(proj)    # It takes the projection and put it onto the new dataset.
        gdal.RasterizeLayer(target_ds, [1], layer,burn_values=[file_order])  # Rasterize our new dataset with current class value (veg=1).
        file_order += 1    # Increment the file order by 1 every time of the loop. 
        vec_band = target_ds.GetRasterBand(1) # target_ds is now raster layer containing vector shp file (as our input file dataset).
        # read all pixels of data after adding current shp file value (value = file_order).
        tr_pixels = vec_band.ReadAsArray()  # 2d layer(one layer).
        # Create list for each index of non zero pixels
        n_class_pixels = len(tr_pixels.nonzero()[0])  # n_class_pixels = 30 for veg class (It takes the pixels that is not equal to zero).
        n_class_train = perc_train_ratio * n_class_pixels / 100  # n_class_train = 21 pixels for veg class.
        n_class_test = n_class_pixels - n_class_train  # n_class_test = 9 pixels for veg class.
        print('\t\tDividing Class pixels into Train & Test steps.')
        print('\t\t#Train:#Test ==> ' + str(n_class_train) + ' pixels : ' + str(n_class_test) + ' pixels\n')
        
        print(n_class_pixels,n_class_train)
        
        # Select 21 random indexes from 30 indexes of veg class as train.
        train_set_index = random.sample(range(0, int(n_class_pixels)), int(n_class_train))
        # Select the other 9 indexes of veg class as test.
        test_set_index = [i for i in range(n_class_pixels) if not i in train_set_index]
        # Fill ts_pixels as tr_pixels (now all of them containing 30 pixels).
        ts_pixels = tr_pixels.copy()
        # Get pixels of nonzero value in shp class.
        pixels_loc = tr_pixels.nonzero()
        
        for i in train_set_index:    # For all train indexes set test pixels as empty.
            ts_pixels[pixels_loc[0][i], pixels_loc[1][i]] = 0   # I want to select the test pixels only so put all train pixels with zero.

        for i in test_set_index:    # For all test indexes set train pixels as empty.
            tr_pixels[pixels_loc[0][i], pixels_loc[1][i]] = 0   # I want to select the train pixels only so put all test pixels with zero.

        # Add train pixels to the accumilative labeled train pixels.
        labeled_tr_pixels += tr_pixels  # Save all train data in one place accumilative (veg,water,....).
        # Add test pixels to the accumilative labeled test pixels.
        labeled_ts_pixels += ts_pixels    # Save all test data in one place accumilative (veg,water,....).
        
        # End of loop here.
        
    # Will take the pixels of the train and put it into training samples and the labels which are (1,2,3,4,5) and put it into training labels.
    is_train = np.nonzero(labeled_tr_pixels)
    training_labels = labeled_tr_pixels[is_train]
    labeled_tr_pixels = None
    # After selecting all train indexes, extract train pixels from the dataset.
    training_samples = bands_data[is_train]
    training_samples = training_samples.astype(float)
    # Will take the pixels of the test and put it into testing samples and the labels to put it into testing labels.
    is_test = np.nonzero(labeled_ts_pixels)
    testing_labels = labeled_ts_pixels[is_test]
    labeled_ts_pixels = None
    # After selecting all test indexes, extract test pixels from the dataset.
    testing_samples = bands_data[is_test]
    testing_samples = testing_samples.astype(float)

    return training_samples, training_labels, testing_samples, testing_labels


def save_classification_results(dir_out, order_id, data_file_name, report_json, ci_2d_colored, c_name):    # Function which saves the classification results.
    print('\n\t6. Saving Classification Report ...')
    report_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name + '_classification_' + c_name + '_report.txt'
    report_json_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name + '_classification_' + c_name + '_json_report.json'
    ci_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name + '_classification_' + c_name + '_rgb.png'
    f = open(report_file_path, "w+")    # Opens the report file and if it does not exist it creates the file.
    f.write(json.dumps(report_json, indent=4, separators=("  ", " : "))) # dumps takes dictinoray table and put it into jason file (in one line).
    f.close()    # Closes the file.
    f_json = open(report_json_file_path, "w+")    # Open the file with json format.
    f_json.write(json.dumps(report_json))    # Write into the file with json format.
    f_json.close()    # Close jason file.

    print('\n\t7. Saving Classified Image ...')
    imsave(ci_file_path, ci_2d_colored)
    ci_2d_colored = None


def svm_process(dir_data, dir_shp, dir_out, data_file_name, order_id, c, g, k): # Function which makes support vector machine algorithm.
    print('\nStarting SVM Classification Script [' + data_file_name + ']')

    if k not in ['linear', 'poly', 'rbf', 'sigmoid']:    # Makes kernel parameter equals linear by default if nothing is selected.
        k = 'linear'
    
    data_file_path = dir_data + '/' + data_file_name    # Takes the directory and the file name and opens it.
    
    report_json = {}    # String which will be filled and then putting it in the report file, Dictionary includes all the attribute names and values in a format like a table.
    report_json[json_names[0]] = order_id    # dict[key]=value.
    report_json[json_names[1]] = 'Classification'
    report_json[json_names[2]] = 'SVM (c = ' + str(c) + ', g = ' + str(g) + ', kernel = ' + k + ')'
    report_json[json_names[3]] = len(classes_names)    # jason names=list.
    #report_json[json_names[4]] = str(classes_names)
    print('\t1. Openeing Data File: ' + data_file_path)
    bands_data, geo_transform, proj = open_dataset(data_file_path)    # data file path= name of the image.
    rows, cols, n_bands = bands_data.shape
    print('\tData Source Dimensions: ' + 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(
        n_bands) + '\n')
    report_json[json_names[5]] = 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(n_bands)
    training_samples, training_labels, testing_samples, testing_labels = extract_shapefiles_pixels(dir_shp, bands_data, geo_transform, proj, cols, rows)
    print('\t\tData and Classes are ready for Classification Process', '\n\t3. Sarting Classification ...')
    # Create SVM classifier.
    classifier = SVC(class_weight='balanced', C=float(c), gamma=float(g), kernel=k)
    # Fit using constructed train pixels (samples) and labels.
    print('\t\tFitting Train Pixels ...')
    classifier.fit(training_samples, training_labels)
    # Predict using test pixels (samples).
    print('\t\tPredicting Test Pixels ...')
    predicted_labels = classifier.predict(testing_samples)
    # Calculate confusion matrix and accuracy (required in classification report output).
    print('Generating Confusion Matrix ...')
    aa = metrics.confusion_matrix(testing_labels, predicted_labels)
    print('Computing Accuracy ...')
    q = metrics.accuracy_score(testing_labels, predicted_labels)
    print('\nConfusion Matrix:\n' + str(aa))
    print('\nAccuracy Score: ' + str(100 * q) + ' %')
    report_json[json_names[6]] = str(list(aa.ravel()))
    report_json[json_names[7]] = str(100 * q) + ' %'
    #report_txt = report_txt + '\nConfusion Matrix:\n' + str(aa) + '\nAccuracy Score: ' + str(100 * q) + ' %'
    print('\tEnd of Classification Process ...\n')

    # Predict for all 2d dataset.
    print('\t4. Predicting Whole Data Pixels ...')
    ci_2d = np.empty([rows, cols], dtype=np.uint8)
    for i in range(rows):
        ci_2d[i, :] = classifier.predict(bands_data[i, :, :])
    bands_data = None

    # Create empty colored image (colored classified image).
    ci_2d_colored = np.empty([rows, cols, 3], dtype=np.uint8)
    class_pixels = np.zeros(len(classes_names), dtype=np.uint32)

    print('\n\t5. Generating Classified Image ...')
    for i in range(rows):
        for j in range(cols):
            # Fill colored ci pixel by pixel.
            ci_2d_colored[i, j] = switcher.get(ci_2d[i, j], (255, 255, 255))
            class_pixels[ci_2d[i, j] - 1] += 1

    # Save outputs to the destinations.
    class_info_dict = {}
    class_info_dict['Total'] = str(rows * cols) + ' pixels'
    class_info = "\n\n# of Pixels per Class:\n\tTotal: " + str(rows * cols) + ' pixels'
    for i in range(len(classes_names)):
        class_info = class_info + '\n\t' + classes_names[i] + ': ' + str(class_pixels[i]) + ' pixels (' + str(
            100 * class_pixels[i] / (rows * cols)) + ' %)'
        class_info_dict[classes_names[i]] = str(class_pixels[i]) + ' pixels (' + str(100 * class_pixels[i] / (rows * cols)) + ' %)'

    report_json[json_names[8]] = class_info_dict

    save_classification_results(dir_out, order_id, data_file_name, report_json, ci_2d_colored, 'svm')
    print('\n\n\t\t\tEnd of SVM Classification Script [' + data_file_name + ']')
    return ci_2d


def cd_svm_process(dir_data, dir_shp, dir_out, data_file_name1, data_file_name2, order_id, c, g, k):    # Function which makes change detection using svm algorithm.
    print('\nStarting Change Detection Script [old: ' + data_file_name1 + ', new: ' + data_file_name2 + ']')
    print('\nA. Starting Classification for two images ...')
    ci_first = svm_process(dir_data, dir_shp, dir_out, data_file_name1, order_id, c, g, k)    # First classified image.
    ci_second = svm_process(dir_data, dir_shp, dir_out, data_file_name2, order_id, c, g, k)    # Second classified image.
    print('\nB. Starting Change Detection Process ...')

    report_json = {}    # String which will be filled and then putting it in the report file, Dictionary includes all the attribute names and values in a format like a table.
    report_json[json_names[0]] = order_id    # dict[key]=value.
    report_json[json_names[1]] = 'Change Detection (Difference Image Generation)'
    report_json[json_names[2]] = 'SVM (c = ' + str(c) + ', g = ' + str(g) + ', kernel = ' + k + ')'
    report_json[json_names[9]] = data_file_name1
    report_json[json_names[10]] = data_file_name2
    ci_cd = []
    ci_cd.append(ci_first)
    ci_cd.append(ci_second)
    ci_cd = np.dstack(ci_cd) # It makes stack to make then in a form of numpy array.
    print(ci_cd.shape)
    rows, cols = ci_first.shape
    print(rows, cols)

    di_2d = np.empty([rows, cols], dtype=np.uint8)
    di_2d_colored = np.empty([rows, cols, 3], dtype=np.uint8)
    state_pixels = np.zeros(len(classes_names) ** 2, dtype=np.float32)
    states_names = []

    print('\nC. Creating Difference Image ...')

    for i in range(rows):
        for j in range(cols):
            di_2d[i, j] = (ci_cd[i, j, 0] - 1) * 6 + ci_cd[i, j, 1]
            di_2d_colored[i, j] = switcher2.get(di_2d[i, j], (255, 255, 255))
            state_pixels[di_2d[i, j] - 1] += 1

    print('\nD. Creating list of states ...')

    for i in range(len(classes_names)):
        for j in range(len(classes_names)):
            states_names.append(classes_names[i] + '-' + classes_names[j])

    state_info_dict = {}
    state_info_dict['Total'] = str(rows * cols) + ' pixels'
    state_info = "\n\n# of Pixels per State:\n\tTotal: " + str(rows * cols) + ' pixels'
    for i in range(len(states_names)):
        state_info_dict[states_names[i]] = str(state_pixels[i]) + ' pixels (' + str( 100 * state_pixels[i] / (rows * cols)) + ' %)'

    report_json[json_names[11]] = state_info_dict
    report_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_svm_report.txt'
    report_json_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_svm_json_report.json'
    di_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_svm_rgb.png'
    print('\nE. Saving Change Detection Report (text and json files) ...')
    f = open(report_file_path, "w+")
    f.write(json.dumps(report_json, indent=4, separators=("  ", " : ")))
    f.close()
    f_json = open(report_json_file_path, "w+")
    f_json.write(json.dumps(report_json))
    f_json.close()

    print('\nF. Savting DI Colored Image ...')
    imsave(di_file_path, di_2d_colored)
    di_2d_colored = None

    print('\nG. End of Change Detection Script [old: ' + data_file_name1 + ', new: ' + data_file_name2 + ']')
    return di_2d


def dt_process(dir_data, dir_shp, dir_out, data_file_name, order_id, max_depth):    # Function which makes decision tree process.

    print('\nStarting Decision Tree Classification Script [' + data_file_name + ']')

    data_file_path = dir_data + '/' + data_file_name    # Takes the directory and the file name and opens it.
    report_json = {}    # String which will be filled and then putting it in the report file, Dictionary includes all the attribute names and values in a format like a table.
    report_json[json_names[0]] = order_id    # dict[key]=value.
    report_json[json_names[1]] = 'Classification'
    report_json[json_names[2]] = 'Decision Tree (Max Depth = ' + str(max_depth) + ')'
    report_json[json_names[3]] = len(classes_names)
    #report_json[json_names[4]] = str(classes_names)
    print('\t1. Openeing Data File: ' + data_file_path)

    bands_data, geo_transform, proj = open_dataset(data_file_path)    # data file path= name of the image.
    rows, cols, n_bands = bands_data.shape
    print('\tData Source Dimensions: ' + 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(
        n_bands) + '\n')
    report_json[json_names[5]] = 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(n_bands)

    training_samples, training_labels, testing_samples, testing_labels = extract_shapefiles_pixels(dir_shp, bands_data,
                                                                                                   geo_transform, proj,
                                                                                                   cols, rows)
    print('\t\tData and Classes are ready for Classification Process', '\n\t3. Sarting Classification ...')
    # Create Decision tree classifier.
    classifier = DecisionTreeClassifier(max_depth = int(max_depth), random_state=None)
    # Fit using constructed train pixels (samples) and labels.
    print('\t\tFitting Train Pixels ...')
    classifier.fit(training_samples, training_labels)
    # Predict using test pixels (samples).
    print('\t\tPredicting Test Pixels ...')
    predicted_labels = classifier.predict(testing_samples)
    # Calculate confusion matrix and accuracy (required in classification report output).
    print('Generating Confusion Matrix ...')
    aa = metrics.confusion_matrix(testing_labels, predicted_labels)
    print('Computing Accuracy ...')
    q = metrics.accuracy_score(testing_labels, predicted_labels)
    print('\nConfusion Matrix:\n' + str(aa))
    print('\nAccuracy Score: ' + str(100 * q) + ' %')
    report_json[json_names[6]] = str(list(aa.ravel()))
    report_json[json_names[7]] = str(100 * q) + ' %'
    print('\tEnd of Classification Process ...\n')

    print('\t4. Predicting Whole Data Pixels ...')
    ci_2d = np.empty([rows, cols], dtype=np.uint8)
    for i in range(rows):
        ci_2d[i, :] = classifier.predict(bands_data[i, :, :])
    bands_data = None
    ci_2d_colored = np.empty([rows, cols, 3], dtype=np.uint8)
    class_pixels = np.zeros(len(classes_names), dtype=np.uint32)

    print('\n\t5. Generating Classified Image ...')
    for i in range(rows):
        for j in range(cols):
            # Fill colored classified image pixel by pixel.
            ci_2d_colored[i, j] = switcher.get(ci_2d[i, j], (255, 255, 255))
            class_pixels[ci_2d[i, j] - 1] += 1

    class_info_dict = {}
    class_info_dict['Total'] = str(rows * cols) + ' pixels'
    for i in range(len(classes_names)):
        class_info_dict[classes_names[i]] = str(class_pixels[i]) + ' pixels (' + str(100 * class_pixels[i] / (rows * cols)) + ' %)'

    report_json[json_names[8]] = class_info_dict

    save_classification_results(dir_out, order_id, data_file_name, report_json, ci_2d_colored, 'dt')
    print('\n\n\t\t\tEnd of Decision Tree Classification Script [' + data_file_name + ']')
    return ci_2d


def cd_dt_process(dir_data, dir_shp, dir_out, data_file_name1, data_file_name2, order_id, max_depth):    # Function which makes change detection for decision tree algorithm.
    print('\nStarting Change Detection Script [old: ' + data_file_name1 + ', new: ' + data_file_name2 + ']')
    print('\nA. Starting Classification for two images ...')
    ci_first = dt_process(dir_data, dir_shp, dir_out, data_file_name1, order_id, max_depth)        # First classified image.
    ci_second = dt_process(dir_data, dir_shp, dir_out, data_file_name2, order_id, max_depth)    # Second classified image.
    print('\nB. Starting Change Detection Process ...')
    report_json = {}    # String which will be filled and then putting it in the report file, Dictionary includes all the attribute names and values in a format like a table.
    report_json[json_names[0]] = order_id    # dict[key]=value.
    report_json[json_names[1]] = 'Change Detection (Difference Image Generation)'
    report_json[json_names[2]] = 'Decision Tree (Max Depth = ' + str(max_depth) + ')'
    report_json[json_names[9]] = data_file_name1
    report_json[json_names[10]] = data_file_name2
    ci_cd = []
    ci_cd.append(ci_first)
    ci_cd.append(ci_second)
    ci_cd = np.dstack(ci_cd) # It makes stack to make then in a form of numpy array.
    print(ci_cd.shape)
    rows, cols = ci_first.shape
    # print(rows, cols)

    di_2d = np.empty([rows, cols], dtype=np.uint8)
    di_2d_colored = np.empty([rows, cols, 3], dtype=np.uint8)
    state_pixels = np.zeros(len(classes_names) ** 2, dtype=np.float32)
    states_names = []
    
    # state_pixels = np.zeros(len(np.unique(di_2d)), dtype=np.float32)
    # states_names = []
    print('\nC. Creating Difference Image ...')
    #print(len(state_pixels),len(classes_names))

    for i in range(rows):
        for j in range(cols):
            di_2d[i, j] = (ci_cd[i, j, 0] - 1) * 6 + ci_cd[i, j, 1]
            di_2d_colored[i, j] = switcher2.get(di_2d[i, j], (255, 255, 255))
            state_pixels[di_2d[i, j] - 1] += 1

    print('\nD. Creating list of states ...')

    for i in range(len(classes_names)):
        for j in range(len(classes_names)):
            states_names.append(classes_names[i] + '-' + classes_names[j])

    state_info_dict = {}
    state_info_dict['Total'] = str(rows * cols) + ' pixels'
    state_info = "\n\n# of Pixels per State:\n\tTotal: " + str(rows * cols) + ' pixels'
    for i in range(len(states_names)):
        state_info_dict[states_names[i]] = str(state_pixels[i]) + ' pixels (' + str( 100 * state_pixels[i] / (rows * cols)) + ' %)'

    report_json[json_names[11]] = state_info_dict
    report_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_dt_report.txt'
    report_json_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_dt_json_report.json'
    di_file_path = dir_out + '/' + 'ORD' + order_id + '_' + data_file_name1 + '_' + data_file_name2 + '_change_dt_rgb.png'

    print('\nE. Saving Change Detection Report (text and json files) ...')
    f = open(report_file_path, "w+")
    f.write(json.dumps(report_json, indent=4, separators=("  ", " : ")))
    f.close()
    f_json = open(report_json_file_path, "w+")
    f_json.write(json.dumps(report_json))
    f_json.close()

    print('\nF. Savting DI Colored Image ...')
    imsave(di_file_path, di_2d_colored)
    di_2d_colored = None

    print('\nG. End of Change Detection Script [old: ' + data_file_name1 + ', new: ' + data_file_name2 + ']')
    return di_2d


def kmeans_process(dir_data, dir_shp, dir_out, data_file_name, order_id, num_of_classes):
    print('\nStarting Unsupervised KMeans Classification Script [' + data_file_name + ']')

    data_file_path = dir_data + '/' + data_file_name    # Takes the directory and the file name and opens it.
    report_json = {}    # String which will be filled and then putting it in the report file, Dictionary includes all the attribute names and values in a format like a table.
    report_json[json_names[0]] = order_id    # dict[key]=value.
    report_json[json_names[1]] = 'Classification'
    report_json[json_names[2]] = 'Unsupervised KMeans (# of classes = ' + str(num_of_classes) + ')'
    report_json[json_names[3]] = num_of_classes
    #report_json[json_names[4]] = str(classes_names_unsupervised[0:num_of_classes+1])
    print('\t1. Openeing Data File: ' + data_file_path)

    bands_data, geo_transform, proj = open_dataset(data_file_path)
    rows, cols, n_bands = bands_data.shape
    print('\tData Source Dimensions: ' + 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(
        n_bands) + '\n')
    report_json[json_names[5]] = 'Rows: ' + str(rows) + ', Columns: ' + str(cols) + ', Bands: ' + str(n_bands)

    training_samples, training_labels, testing_samples, testing_labels = extract_shapefiles_pixels(dir_shp, bands_data,
                                                                                                   geo_transform, proj,
                                                                                                   cols, rows)
    print('\t\tData and Classes are ready for Classification Process', '\n\t3. Sarting Classification ...')
    # Create k means classifier.
    classifier = KMeans(n_clusters=int(num_of_classes))
    # Fit using constructed train pixels (samples) and labels.
    print('\t\tFitting Train Pixels ...')
    classifier.fit(training_samples)
    # Predict using test pixels (samples).
    print('\t\tPredicting Test Pixels ...')
    predicted_labels = classifier.predict(testing_samples)
    # Calculate confusion matrix and accuracy (required in classification report output).
    print('Generating Confusion Matrix ...')
    aa = metrics.confusion_matrix(testing_labels, predicted_labels)
    print('Computing Accuracy ...')
    q = metrics.accuracy_score(testing_labels, predicted_labels)
    print('\nConfusion Matrix:\n' + str(aa))
    print('\nAccuracy Score: ' + str(100 * q) + ' %')
    report_json[json_names[6]] = str(list(aa.ravel()))
    report_json[json_names[7]] = str(100 * q) + ' %'
    print('\tEnd of Classification Process ...\n')

    print('\t4. Predicting Whole Data Pixels ...')
    ci_2d = np.empty([rows, cols], dtype=np.uint8)
    for i in range(rows):
        ci_2d[i, :] = classifier.predict(bands_data[i, :, :])
    bands_data = None
    ci_2d_colored = np.empty([rows, cols, 3], dtype=np.uint8)
    class_pixels = np.zeros(len(classes_names), dtype=np.uint32)

    print('\n\t5. Generating Classified Image ...')
    for i in range(rows):
        for j in range(cols):
            ci_2d_colored[i, j] = switcher_unsupervised.get(ci_2d[i, j], (255, 255, 255))
            class_pixels[ci_2d[i, j] - 1] += 1

    class_info_dict = {}
    class_info_dict['Total'] = str(rows * cols) + ' pixels'
    for i in range(int(num_of_classes)):
        class_info_dict[classes_names_unsupervised[i]] = str(class_pixels[i]) + ' pixels (' + str(100 * class_pixels[i] / (rows * cols)) + ' %)'

    report_json[json_names[8]] = class_info_dict

    save_classification_results(dir_out, order_id, data_file_name, report_json, ci_2d_colored, 'kmeans')
    print('\n\n\t\t\tEnd of Unsupervised KMeans Classification Script [' + data_file_name + ']')
    return ci_2d
