#ifndef __MULTILAYER_STIXEL_WORLD_H__
#define __MULTILAYER_STIXEL_WORLD_H__

#include <opencv2/core.hpp>

/** @brief Stixel geometric class id
*/
enum
{
	GEOMETRIC_ID_GROUND = 0,
	GEOMETRIC_ID_OBJECT = 1,
	GEOMETRIC_ID_SKY = 2,
};

/** @brief Stixel struct
*/
struct Stixel
{
	int uL;                   //!< stixel left x position
	int vT;                   //!< stixel top y position
	int vB;                   //!< stixel bottom y position
	int width;                //!< stixel width
	int geoId;                //!< stixel geometric class id
	int semId;                //!< stixel semantic class id
	cv::Vec2f disp;           //!< stixel disparity function (slope and intercept)
};

/** @brief CameraParameters struct
*/
struct CameraParameters
{
	float fu;                 //!< focal length x (pixel)
	float fv;                 //!< focal length y (pixel)
	float u0;                 //!< principal point x (pixel)
	float v0;                 //!< principal point y (pixel)
	float baseline;           //!< baseline (meter)
	float height;             //!< height position (meter)
	float tilt;               //!< tilt angle (radian)

	// default settings
	CameraParameters()
	{
		fu = 1.f;
		fv = 1.f;
		u0 = 0.f;
		v0 = 0.f;
		baseline = 0.2f;
		height = 1.f;
		tilt = 0.f;
	}
};

/** @brief SlantedStixels class.

The class implements the Slanted Stixel computation based on [1][2][3].
[1] Hernandez-Juarez, Daniel, et al. "Slanted Stixels: A way to represent steep streets." International Journal of Computer Vision 127.11 (2019): 1643-1658.
[2] Hernandez-Juarez, Daniel, et al. "3D Perception with Slanted Stixels on GPU." IEEE Transactions on Parallel and Distributed Systems (2021).
*/
class SlantedStixels
{
public:

	enum
	{
		STIXEL_WIDTH_4 = 4, //!< stixel width
		STIXEL_WIDTH_8 = 8  //!< stixel width
	};

	enum
	{
		STIXEL_Y_RESOLUTION_4 = 4, //!< stixel vertical resolution
		STIXEL_Y_RESOLUTION_8 = 8  //!< stixel vertical resolution
	};

	/** @brief Parameters struct
	*/
	struct Parameters
	{
		// disparity range
		int dmin;
		int dmax;

		// stixel width
		int stixelWidth;

		// stixel vertical resolution
		int stixelYResolution;

		// camera parameters
		CameraParameters camera;

		// geometry id for each class
		std::vector<int> geometry;

		// default settings
		Parameters()
		{
			// disparity range
			dmin = 0;
			dmax = 64;

			// stixel width
			stixelWidth = STIXEL_WIDTH_4;

			// stixel vertical resolution
			stixelYResolution = STIXEL_Y_RESOLUTION_4;

			// camera parameters
			camera = CameraParameters();
		}
	};

	/** @brief Creates an instance of SlantedStixels.
		@param param Input parameters.
	*/
	static cv::Ptr<SlantedStixels> create(const Parameters& param = Parameters());

	/** @brief Computes slanted stixels from a disparity map and a disparity confidence.
		@param disparity Input 32-bit disparity map.
		@param confidence Input disparity confidence of the same size and the same type as disparity.
		@param stixels Output array of stixels.
	*/
	virtual void compute(const cv::Mat& disparity, const cv::Mat& confidence,
		std::vector<Stixel>& stixels) = 0;

	/** @brief Computes slanted stixels from a disparity map and a disparity confidence.
		@param disparity Input 32-bit disparity map.
		@param confidence Input disparity confidence of the same size and the same type as disparity.
		@param predict Input 32-bit 3-dimensional semantic segmentation scores.
		@param stixels Output array of stixels.
	*/
	virtual void compute(const cv::Mat& disparity, const cv::Mat& confidence, const cv::Mat& predict,
		std::vector<Stixel>& stixels) = 0;

	/** @brief Sets parameters to SlantedStixels.
		@param param Input parameters.
	*/
	virtual void setParameters(const Parameters& param) = 0;
};

#endif // !__SLANTED_STIXELS_H__
