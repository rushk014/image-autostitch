<h1 align="center"> Autostitching Panoramic Images </h1>

<p align="center">Python application for stitching panoramic images.</p>

## Description

Implements stitching arbitrary images either through manually defined features
or through SIFT feature matching based on the Harris corner detector, as described in [Brown et Al.](http://matthewalunbrown.com/papers/cvpr05.pdf) Stitching is accomplished using RANSAC to compute a robust homography between the set(s) of image
correspondence, then warping each image toward a chosen reference image. Finally, multiresolution blending is accomplished using 
Laplacian pyramids. The application also supports image rectification.

## Usage

Requirements are listed in ```requirements.txt```. Application is run from ```main.py```: 

```bash
usage: main.py [-h] -r REF [-w WARP [WARP ...]] -m
               {rectify,manual_mosaic,autostitch} [-v]

optional arguments:
  -h, --help            show this help message and exit
  -r REF, --ref REF     reference image path
  -w WARP [WARP ...], --warp WARP [WARP ...]
                        warp image path(s)
  -m {rectify,manual_mosaic,autostitch}, --mode {rectify,manual_mosaic,autostitch}
                        choose between image rectification, manual
                        correspondence mosaicing and autostitched mosaicing
  -v, --verbose         log file reads/writes to stdout & visualize
                        intermediate autostitching outputs
```

### Modes

- rectify: requires only ```-r REF```
- manual_mosaic: requires ```-r REF``` and at least one ```-w WARP```, supports a single global feature set
- autostitching: requires ```-r REF``` and at least one ```-w WARP```

Global filepaths for intermediate and final outputs are set in ```utils.py```. 

Note: Click same coordinate twice to end manual shape labelling.

## Examples

### Rectification

<table>
  <tr>
     <td>Hallway</td>
     <td>Hallway Rectified</td>
  </tr>
  <tr>
    <td><img alt="hallway" src="_examples/hallway.jpg" width="325"/> </td>
    <td><img alt="hallway rectified" src="_examples/hallway_rectified.jpg" width="325"/></td>
  </tr>
 </table>



### Mosaicing

<table>
  <tr>
     <td>Building Left</td>
     <td>Building Right</td>
     <td>Building Mosaic</td>
  </tr>
  <tr>
    <td><img alt="building-left" src="_examples/building-1.jpg" width="325"/></td>
    <td><img alt="building-right" src="_examples/building-2.jpg" width="325"/></td>
    <td><img alt="building mosaic" src="_examples/building-mosaic.jpg" width="375"/></td>
  </tr>
  <tr>
     <td>Room Left</td>
     <td>Room Right</td>
     <td>Room Mosaic</td>
  </tr>
  <tr>
    <td><img alt="room-left" src="_examples/room-1.jpg" width="325"/></td>
    <td><img alt="room-right" src="_examples/room-2.jpg" width="325"/></td>
    <td><img alt="room mosaic" src="_examples/room-mosaic.jpg" width="375"/></td>
  </tr>
 </table>

 
