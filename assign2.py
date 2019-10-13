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
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    x = np.arange(-5,6)
    filter = np.exp((x**2)/-2/(sigma**2))
    filter = filter[1:-1]

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    result = convolve1d(img,filter,1,np.float64,'constant')

    all_values_1 = img
    all_values_1.fill(1)
    weight = convolve1d(all_values_1,filter,1,np.float64,'constant')
    img_smoothed = result/weight
    return img_smoothed
################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(img,sigma)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(img_smoothed.T,sigma)
    img_smoothed = img_smoothed.T
    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    #  TODO: compute Ix & Iy
    Ix = img
    Iy = img
    #computing Ix
    for y in range(img.shape[0]):
        for x in range(1,img.shape[1]-1): # central difference for non-border case
            Ix[y][x]=(img[y][x+1]-img[y][x-1])/2
        Ix[y][0] = img[y][1]-img[y][0] # forward difference for x=0 border case
        Ix[y][img.shape[1]-1] = img[y][img.shape[1]-1] - img[y][img.shape[1]-2] # backward difference for x=w-1 border case

    for x in range(img.shape[1]):
        for y in range(1,img.shape[0]-1):
            Iy[y][x]=(img[y+1][x]-img[y-1][x])/2
        Iy[0][x] = img[1][x]-img[0][x]
        Iy[y][img.shape[0]-1] = img[img.shape[0]-1][x] - img[img.shape[0]-2][x]

    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = np.multiply(Ix,Ix)
    Iy2 = np.multiply(Iy,Iy)
    IxIy = np.multiply(Ix,Iy)
    # TODO: smooth the squared derivatives
    smoothed_Ix2 = smooth2D(Ix2,sigma)
    smoothed_Iy2 = smooth2D(Iy2, sigma)
    smoothed_IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    R = (np.multiply(Ix2,Iy2) - np.multiply(IxIy,IxIy)) - 0.04*(np.multiply(Ix2+Iy2,Ix2+Iy2))
    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    # TODO: perform thresholding and discard weak corners

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # img = img_gray
    # Ix = img
    # Iy = img
    # for y in range(img.shape[0]):
    #     for x in range(1,img.shape[1]-1): # central difference for non-border case
    #         Ix[y][x]=(img[y][x+1]-img[y][x-1])/2
    #     Ix[y][0] = img[y][1]-img[y][0] # forward difference for x=0 border case
    #     Ix[y][img.shape[1]-1] = img[y][img.shape[1]-1] - img[y][img.shape[1]-2] # backward difference for x=w-1 border case
    #
    # for x in range(img.shape[1]):
    #     for y in range(1,img.shape[0]-1):
    #         Iy[y][x]=(img[y+1][x]-img[y-1][x])/2
    #     Iy[0][x] = img[1][x]-img[0][x]
    #     Iy[y][img.shape[0]-1] = img[img.shape[0]-1][x] - img[img.shape[0]-2][x]
    #
    # Ix2 = np.multiply(Ix,Ix)
    # print(type(Ix2))
    # print(Ix2.shape)
    # plt.imshow(np.float32(Ix), cmap = 'gray')
    # plt.show()

    img_smoothed = smooth2D(img_gray,1)
    # plt.imshow(np.float32(img_smoothed),cmap = 'gray')
    # plt.show()
    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
