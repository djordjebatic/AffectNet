#!/usr/bin/python

import keras.backend as K
import sys
from deployment.tensorflow_detector import *
from deployment.utils import label_map_util
from deployment.utils import visualization_utils_color as vis_util
from deployment.video_threading_optimization import *

## OPTIONS ##

PATH_TO_CKPT = './frozen_graphs/frozen_inference_graph_face.pb'
PATH_TO_CLASS = './frozen_graphs/classificator_full_model.pb'
PATH_TO_REGRESS = './frozen_graphs/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('./protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def predict_from_camera(detector):
    print('Press q to exit')
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    window_not_set = True

    while True:
        # grab the frame from the threaded video stream
        image = vs.read()
        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)

        boxes, scores, classes, num_detections, emotions_print = detector.run(image)

        text = "classes: {}".format(emotions_print)
        cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.35, color=(0, 255, 0))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1)

        if window_not_set is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            window_not_set = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def predict_from_video(detector, file_path):
    vs = FileVideoStream(file_path).start()
    time.sleep(2.0)
    time.sleep(2.0)
    fps = FPS().start()
    out = None
    frame_count = 0

    while vs.more():
        # grab the frame from the threaded video stream
        image = vs.read()
        frame_count += 1

        if out is None:
            [h, w] = image.shape[:2]
            out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))

        # Check if this is the frame closest to 5 seconds
        if frame_count == 2:
            frame_count = 0

            boxes, scores, classes, num_detections, emotions = detector.run(image)

            text = "classes: {}".format(emotions)
            cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.35, color=(0, 255, 0))

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1)

        out.write(image)
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":

    if sys.argv[1] == '-c':
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
        predict_from_camera(detector)

    elif sys.argv[1] == '-v':
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
        predict_from_video(detector, 'test.mp4')

    else:
        print('Wrong argument')