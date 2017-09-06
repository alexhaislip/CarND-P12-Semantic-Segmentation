import os
import sys
from main import *
from moviepy.editor import VideoFileClip
import collections as cl
import scipy

num_classes = 2
epochs = 10
batch_size = 8
data_dir = './data'
runs_dir = './runs'
scale = 0.75
tests.test_for_kitti_dataset(data_dir)

# Download pretrained vgg model
helper.maybe_download_pretrained_vgg(data_dir)
vgg_path = os.path.join(data_dir, 'vgg')

# input_file = sys.argv[1]
input_file = '/Users/tantony/Movies/Dashcam/2017_0820_153800_447_FancyCars.MP4'
weights_path = './data/ckpt/model_40epoch_0050_final.ckpt'
output_file = './output.mp4'

image_shape = tuple(int(w*scale) for w in clip.size)

tf.reset_default_graph()
with tf.Session() as sess:
    image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
    nn_layers = layers(layer3_out, layer4_out, layer7_out, num_classes)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    correct_label = tf.placeholder(tf.float32, shape=(None, None, None, num_classes), name='correct_label')
    logits, train_op, cross_entropy_loss = optimize(nn_layers, correct_label, learning_rate, num_classes)

    saver = tf.train.Saver()
    saver.restore(sess, weights_path)

    clip = VideoFileClip(input_file).subclip(0,10)

    history = cl.deque(maxlen=10)
    def get_mask(frame):
        image = scipy.misc.imresize(frame, image_shape)
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_input: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        history.append(street_im)
        n = len(history)
        return sum(im/n for im in history)

    clip.fl_image(get_mask)
    clip.write_videofile(output_file, audio=False)
