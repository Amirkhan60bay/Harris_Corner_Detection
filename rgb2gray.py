################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage
from scipy import misc
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :


    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    y_model = np.array([0.299, 0.587, 0.115], dtype=np.float64)
    img_gray = img_color @ y_model
    img_gray.fill(1)
    return img_gray

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
#     parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
#     parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
#     parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
#     print('sigma      : %.2f' % args.sigma)
#     print('threshold  : %.2e' % args.threshold)
#     print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
#         img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
#     plt.imshow(np.uint8(img_color))
#     plt.show()
#     print(img_color[20,10])
#     print(type(img_color))

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    main()