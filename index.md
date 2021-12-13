# Autostitching Panoramic Images

## Overview

In this project, I implement an algorithm to stitch arbitrary images either through manually defined features correspondences or using SIFT feature matching based on the Harris corner detector, as described in [Brown et Al.](http://matthewalunbrown.com/papers/cvpr05.pdf). In the case of SIFT features, the final stitching is computed using RANSAC to find a robust homography between the set(s) of image correspondences. Once the homography is computed, each image is warped toward a chosen reference image. Finally, multiresolution blending is accomplished using Laplacian pyramids. The application also supports image rectification using a manually defined 4-point correspondence.

## Image Warping and Mosaicing

### Shooting the Pictures

I used a DSLR with manual exposure and focus locking, as well as a tripod, to shoot several photos with significant field of view overlap, taken from a fixed camera position. Supplemental photos were taken using my iPhone with AE/AF locking enabled. 

### Image Rectification

![hallway](/assets/imgs/readme/hallway.jpg "Hallway")

Using the image of a hallway, I manually defined correspondence around a doorway and a mural, both nearly perpendicular to the camera's direction vector, then computed the homography to the unit rectangle `[[0, 0], [w, 0], [0, h], [w, h]]`.

![hallway-rectified](/assets/imgs/readme/hallway_rectified.jpg "Hallway Rectified")


### Recovering Homographies and Image Warping

I first manually defined the correspondences between the two images, or in the case of image rectification,
one image and the unit rectangle. Significantly overlapping field of view helps with defining image correspondences, increasing the number of shared features to define correspondences upon.

| ![room-1](/assets/imgs/docs/room-1_vec.jpg "Room-1") | ![room-2](/assets/imgs/docs/room-2_vec.jpg "Room-2")
|:--:|:--:|
| *Room-1* | *Room-2* |


### Mosaicing

After defining correspondence between the images, I designate a reference image and warp the other image to it. After placing the images (warped and non-warped) into a empty image, which required specific predictive computations to determine output size, I initial used naive alpha blending to overlap the images, however this produced strong edge artifacts. I later implemented a multiresolution blending algorithm based on Laplacian pyramids to blend the images.

![room-mosaic](/assets/imgs/docs/room-mosaic.jpg "Room Mosaic")

#### Arbitrary Image Mosaicing

Extending my implementation to support an arbitrary number of image vectors required some fine tuning to properly calculate output sizes, but the overall implementation follows the same formula. Currently my implementation only supports one set of correspondences, extending to support an arbitrary number of correspondences between arbitrary images would allow full panoramic, as opposed to the current `~180°` field of view. I warp images taken to the left/right of a user-defined reference image towards the reference image using homographies, then iteratively blend consequetive images using multiresolution blending.

![green-room-mosaic](/assets/imgs/docs/green-room-mosaic.jpg "Green Room Mosaic")

## Feature Matching and Autostitching

### Harris Corners

I used the given starter code `harris.py` to generate harris points for both images. Given the high resolution of modern cameras, I was initially computing upwards of `100,000` harris points per image. Due to the high computational complexity of the ANMS algorithm, I was forced to resize images. Anecdotally the optimal number of harris points seems to be around `10,000` - achieved with a resize fraction of `0.2`.

![room-1-harris](/assets/imgs/docs/room-1_harris.jpg "Room 1 Harris")
![room-2-harris](/docs/assets/imgs/docs/room-2_harris.jpg "Room 2 Harris")

### Adaptive Non-Maximal Suppression

I implemented ANMS by iterating over points, computing the minimum squared distance for each point to another point where `H_point < 0.9 * H_other`. I then sort in the list in descending order and choose the top `500`, as suggested in the paper. Visualized below are the results of ANMS across a variety of source images:

![room-1-anms](/assets/imgs/docs/room-1_anms.jpg "Room 1 ANMS")
![room-2-anms](/assets/imgs/docs/room-2_anms.jpg "Room 2 ANMS")

### Feature Extraction and Matching

To produce feature descriptors for each point in the ANMS output, we extract an axis-aligned `40x40` patch around each feature then downsample it to a `8x8` patch, which is saved as a length `64` vector. In order to match features, we implement Lowe's thresholding technique, using `e1-NN/e2-NN`. Under this technique, we only add features where the ratio of the first and second best distances is less than the threshold, usually set around `0.3`.

![room-1-lowe](/assets/imgs/docs/room-1_lowe.jpg "Room 1 Lowe")
![room-2-lowe](/assets/imgs/docs/room-2_lowe.jpg "Room 2 Lowe")

### RANSAC

To implement RANSAC, I repeatedly subsample a 4-point correspondence to generate a homography, using it to compare the pixels warped under the homography to the correspondending image and updating when squared error is below some threshold, usually around 1 pixel. Finally I take the best correspondence returned after `~1000` iterations of RANSAC and upsample it to fit source image sizes.

![room-1-ransac](/assets/imgs/docs/room-1_RANSAC.jpg "Room 1 RANSAC")
![room-2-ransac](/assets/imgs/docs/room-2_lowe.jpg "Room 2 RANSAC")

## Conclusion

### Mistakes

After implementing RANSAC, I spent several days generating tons of 2-image mosaics, but was consistently unable to tune the thresholds for feature matching and RANSAC to choose correct points. My code was usually finding 2-3 accurate correspondences, but failed to properly match 4 and thus produced an incorrect warp which generated several unintended artifacts and usually resulted in an unusable gradient of pixels. Since my code seemed to almost align the images, I figured my implementation must be mostly correct and there was some error in the kind of resolution or subject matter of the photos I was taking. I spent several days banging my head against the wall trying to figure this out. It was only after Prof. Efros mentioned that his incorrect implementation still produced nearly correct results due to the sheer robustness of the algorithm. Ultimately I discovered my error - sorting the ANMS distances in ascending rather than descending order - and was rewarded with a working autostitching algorithm

### Comparisons

In general, my autostitching code generates mosaics of equal or better quality than manually defined correspondence. There were several instances where the autostitched alignment was far more flush and there were fewer edge artifacts on the autostitched counterpart.

### Results

![room-mosaic](/assets/imgs/readme/room-mosaic.jpg "Room Mosaic")
![building-mosaic](/assets/imgs/readme/building-mosaic.jpg "Building Mosaic")

