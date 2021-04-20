# slanted-stixels
An implementation of Slanted Stixels

<img src=https://github.com/gishi523/slanted-stixels/wiki/images/depth_input.png width=400> <img src=https://github.com/gishi523/slanted-stixels/wiki/images/semantic_input.png width=400>  
<img src=https://github.com/gishi523/slanted-stixels/wiki/images/slanted_stixels_depth.png width=400> <img src=https://github.com/gishi523/slanted-stixels/wiki/images/slanted_stixels_semantic.png width=400>

## Description
- An implementation of the slanted stixel computation based on [1][2]
	- Extracts slanted stixels from a disparity map, a disparity confidence, and a semantic segmentation
	- Jointly infers geometric and semantic layout of traffic scenes
- For disparity confidence, the Local Curve (LC) is implemented based on [3]
- For semantic segmentation, OpenCV DNN module and [ERFNet](https://github.com/Eromera/erfnet_pytorch) is used

## References
- [1] Hernandez-Juarez, D., Schneider, L., Cebrian, P., Espinosa, A., Vazquez, D., López, A. M., ... & Moure, J. C. (2019). Slanted Stixels: A way to represent steep streets. International Journal of Computer Vision, 127(11), 1643-1658.
- [2] Hernandez-Juarez, D., Espinosa, A., Vazquez, D., Lopez, A. M., & Moure, J. C. (2021). 3D Perception with Slanted Stixels on GPU. IEEE Transactions on Parallel and Distributed Systems.
- [3] Pfeiffer, D., Gehrig, S., & Schneider, N. (2013). Exploiting the power of stereo confidences. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 297-304).

## Requirement
- OpenCV (recommended latest version)
- OpenMP (optional)

## How to build
```
$ git clone https://github.com/gishi523/slanted-stixels.git
$ cd slanted-stixels
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
### Command-line arguments
```
Usage: slanted_stixels [params] image-format-L image-format-R
	-h, --help
		print help message.
	image-format-L
		input left image sequence.
	image-format-R
		input right image sequence.
	--camera
		path to camera parameters.
	--start-number (value:1)
	    start frame number.
	--model
	    path to a binary file of model contains trained weights.
	--classes
	    path to a text file with names of classes.
	--colors
	    path to a text file with colors for each class.
	--geometry
	    path to a text file with geometry id (0:ground 1:object 2:sky) for each class.
	--width (value:1024)
		input image width for neural network.
	--height (value:512)
	    input image height for neural network.
	--backend (value:0)
		computation backend. see cv::dnn::Net::setPreferableBackend.
	--target (value:0)
	    target device. see cv::dnn::Net::setPreferableTarget.
	--depth-only
		compute without semantic segmentation.
```

### Example
```
cd slanted_stixels

./build/slanted_stixels \
path_to_left_images/stuttgart_00_000000_%06d_leftImg8bit.png \
path_to_right_images/stuttgart_00_000000_%06d_rightImg8bit.png \
--camera=camera_parameters/cityscapes.xml \
--model=etfnet/erfnet.onnx \
--classes=etfnet/classes.txt \
--colors=etfnet/colors.txt \
--geometry=etfnet/geometry.txt \
--target=1
```

If you have manually built OpenCV DNN module with CUDA backend,
you can pass `DNN_BACKEND_CUDA(=5)` and `DNN_TARGET_CUDA(=6)` to run the semantic segmentation faster.
```
--backend=5 --target=6
```

With `--depth-only` argument, you can test slanted stixel computation with depth information only.

```
cd slanted_stixels

./build/slanted_stixels \

path_to_left_images/imgleft%09d.pgm \
path_to_right_images/imgright%09d.pgm \
--camera=camera_parameters/daimler_urban_seg.xml \
--depth-only
```
