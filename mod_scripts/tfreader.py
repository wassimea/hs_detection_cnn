import tensorflow as tf
import numpy as np
import cv2
#import matplotlib.pyplot as plt

x = tf.python_io.tf_record_iterator(r'/Data2TB/correctly_registered/augmented/train.record')
for example in tf.python_io.tf_record_iterator(r'/Data2TB/correctly_registered/augmented/train.record'):
    result = tf.train.Example.FromString(example)
    filename = result.features.feature['image/filename'].bytes_list.value
    f = str(filename)
    if "-resized-1-rotated-0.png" in f:
        xmins = result.features.feature['image/object/bbox/xmin'].float_list.value
        xmaxs = result.features.feature['image/object/bbox/xmax'].float_list.value
        ymins = result.features.feature['image/object/bbox/ymin'].float_list.value
        ymaxs = result.features.feature['image/object/bbox/ymax'].float_list.value
        height = result.features.feature['image/height'].int64_list.value[0]
        width = result.features.feature['image/height'].int64_list.value[0]
        labels = result.features.feature['image/object/class/label'].int64_list.value
        encoded_image = result.features.feature['image/encoded'].bytes_list.value[0]
        z=xmins[0]
        decoded_image = tf.image.decode_image(encoded_image, channels=3)
        sess = tf.Session()
        decoded_image = sess.run(decoded_image)

        for i in range(0,len(xmins)):
            xmin = int(xmins[i] * width)
            xmax = int(xmaxs[i] * width)
            ymin = int(ymins[i] * height)
            ymax = int(ymaxs[i] * height)
            #label = labels[i]
            if labels[i] == 1:
                cv2.rectangle(decoded_image,(xmin,ymin),(xmax ,ymax),(0,0,255),3)
            else:
                #cv2.rectangle(decoded_image,(xmin,ymin),(xmax ,ymax),(255,0,0),3)
                cv2.circle(decoded_image,(ymin, xmin), 5, (255,0,0), -1)
        x = 1
        cv2.imshow("ayre",decoded_image)
        cv2.waitKey()



    z = 1
    #for item in result.features.feature['image/temp_values'].int64_list.value:
        #with open('C:\\Users\\welah\\Desktop\\test\\txt.txt', 'a') as the_file:
            #the_file.write(str(item) + "\n")
