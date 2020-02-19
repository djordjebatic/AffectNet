import time

import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet import preprocess_input

emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']


class TensorflowDetector(object):
    def __init__(self, PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS):
        """
            Tensorflow detector
        """

        # face detector
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            '''
                device_count={"CPU": 4},
                inter_op_parallelism_threads=4,
                intra_op_parallelism_threads=1
            '''
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_1 = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

        # classificator
        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CLASS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.classification_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_2 = tf.compat.v1.Session(graph=self.classification_graph, config=config)

        # regressor
        self.regression_graph = tf.Graph()
        with self.regression_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_REGRESS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.regression_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_3 = tf.compat.v1.Session(graph=self.regression_graph, config=config)

    def run(self, image):
        """image: bgr image
        return boxes, scores, classes, num_detections, emotions
        """

        [h, w] = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Detection inference
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess_1.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        print('Detection time: {}'.format(round(time.time() - start_time, 8)))
        print('--------------------------------------')

        classification_input = self.classification_graph.get_tensor_by_name('input_1:0')
        classification_output = self.classification_graph.get_tensor_by_name('dense_2/Softmax:0')
        regression_input = self.regression_graph.get_tensor_by_name('input_1:0')
        regression_output = self.regression_graph.get_tensor_by_name('dense_2/BiasAdd:0')

        # this array will be fed to classificator and regressor if image
        # contains any face that is detected with over 0.7 confidence
        images_for_prediction = []
        for i in range(min(20, np.squeeze(boxes).shape[0])):
            if scores is None or np.squeeze(scores)[i] > 0.7:
                ymin, xmin, ymax, xmax = np.squeeze(boxes)[i]
                # making the bounding box a bit bigger, similar to training data
                image_pred = image[max(int(h * ymin) - 20, 0):min(int(h * ymax) + 20, image.shape[:2][0]),
                             max(int(w * xmin) - 20, 0):min(int(w * xmax) + 20, image.shape[:2][1]), :]
                # converting numpy image to PIL
                image_pred = Image.fromarray(image_pred).resize((224, 224))
                image_pred = keras.preprocessing.image.img_to_array(image_pred)
                image_pred = preprocess_input(image_pred)
                images_for_prediction.append(image_pred)

        emotions_detected = []
        if len(images_for_prediction) > 0:
            start_time = time.time()
            prediction = self.sess_2.run(classification_output,
                                         feed_dict={classification_input: images_for_prediction})
            print('Classification time: {}'.format(round(time.time() - start_time, 8)))
            # prints classification class and confidence score for each detected face
            for row in prediction:
                pred = np.argmax(row)
                print(emotion_classes[pred] + ' ' + str(round(row[pred], 2)))
                emotions_detected.append(emotion_classes[pred])
            print('--------------------------------------')
            start_time = time.time()
            prediction = self.sess_3.run(regression_output,
                                         feed_dict={regression_input: images_for_prediction})
            print('Regression time: {}'.format(round(time.time() - start_time, 8)))
            # prints valence and arousal score for each detected face
            for row in prediction:
                print('Valence: ' + str(round(row[0], 5)) + ' Arousal: ' + str(round(row[1], 5)))

            print('\n')

        return boxes, scores, classes, num_detections, emotions_detected
