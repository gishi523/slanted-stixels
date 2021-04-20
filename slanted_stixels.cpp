#include "slanted_stixels.h"

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for schedule(dynamic))
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////

// geometric class id
static constexpr int G = GEOMETRIC_ID_GROUND;
static constexpr int O = GEOMETRIC_ID_OBJECT;
static constexpr int S = GEOMETRIC_ID_SKY;
static constexpr int GEO_ID_BIT = 2;

// maximum cost
static constexpr float Cinf = std::numeric_limits<float>::max();

// model complexity prior
static constexpr float Cmc = 4;

// structural priors
static constexpr float alphaGapP  = 1;
static constexpr float betaGapP   = 0;
static constexpr float alphaGapN  = 1;
static constexpr float betaGapN   = 0;
static constexpr float alphaGravP = 1;
static constexpr float betaGravP  = 1;
static constexpr float alphaGravN = 1;
static constexpr float betaGravN  = 1;
static constexpr float alphaOrd   = 1;
static constexpr float betaOrd    = 1;

// disparity measurement uncertainty
static constexpr float sigmaD[3] =
{
	1.f,
	1.f,
	2.f
};

static constexpr float sigmaDSq[3] =
{
	sigmaD[G] * sigmaD[G],
	sigmaD[O] * sigmaD[O],
	sigmaD[S] * sigmaD[S]
};

static constexpr float invSigmaDSq[3] =
{
	1.f / sigmaDSq[G],
	1.f / sigmaDSq[O],
	1.f / sigmaDSq[S]
};

// range of depth into witch objects are allowed to extend
static constexpr float deltaZ = 0.3f;

// sigma for plane prior
static constexpr float sigma_aG = 1.f;
static constexpr float sigma_bG = 1.f;
static constexpr float sigmaSq_aG = sigma_aG * sigma_aG;
static constexpr float sigmaSq_bG = sigma_bG * sigma_bG;

// camera height and tilt uncertainty
constexpr float sigmaH = 0.05f;
constexpr float sigmaA = 0.005f;
constexpr float sigmaHSq = sigmaH * sigmaH;
constexpr float sigmaASq = sigmaA * sigmaA;

// semantic cost weight
static constexpr float wsem = 0.5f;

// default cost
static constexpr float Cdef = 0;

////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////

struct Line
{
	Line(float a = 0, float b = 0) : a(a), b(b) {}
	Line(const cv::Vec2f& vec) : a(vec[0]), b(vec[1]) {}
	Line(const cv::Point2f& pt1, const cv::Point2f& pt2)
	{
		a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		b = -a * pt1.x + pt1.y;
	}
	inline float operator()(int x) const { return a * x + b; }
	float vhor() const { return -b / a; }
	float a, b;
};

struct PlaneSAT
{
	PlaneSAT(int n) : sxx(0), sxy(0), syy(0), sx(0), sy(0), sw(0)
	{
		SATxx.resize(n);
		SATxy.resize(n);
		SATyy.resize(n);
		SATx.resize(n);
		SATy.resize(n);
		SATw.resize(n);
	}

	inline void add(int v, float d, float w)
	{
		const float x = static_cast<float>(v);
		const float y = d;

		sxx += w * x * x;
		sxy += w * x * y;
		syy += w * y * y;
		sx += w * x;
		sy += w * y;
		sw += w;

		SATxx[v] = sxx;
		SATxy[v] = sxy;
		SATyy[v] = syy;
		SATx[v] = sx;
		SATy[v] = sy;
		SATw[v] = sw;
	}

	inline void setInterval(int vB)
	{
		sxx = SATxx[vB];
		sxy = SATxy[vB];
		syy = SATyy[vB];
		sx = SATx[vB];
		sy = SATy[vB];
		sw = SATw[vB];
	}

	inline void setInterval(int vB, int vT)
	{
		sxx = SATxx[vB] - SATxx[vT - 1];
		sxy = SATxy[vB] - SATxy[vT - 1];
		syy = SATyy[vB] - SATyy[vT - 1];
		sx = SATx[vB] - SATx[vT - 1];
		sy = SATy[vB] - SATy[vT - 1];
		sw = SATw[vB] - SATw[vT - 1];
	}

	std::vector<float> SATxx, SATxy, SATyy, SATx, SATy, SATw;
	float sxx, sxy, syy, sx, sy, sw;
};

////////////////////////////////////////////////////////////////////////////////////
// Static functions
////////////////////////////////////////////////////////////////////////////////////

static constexpr float squared(float x)
{
	return x * x;
}

static cv::Mat1f getch(const cv::Mat1f& src, int id)
{
	return cv::Mat1f(src.size[1], src.size[2], (float*)src.ptr<float>(id));
}

static void create3d(cv::Mat1f& mat, int size0, int size1, int size2)
{
	const int sizes[3] = { size0, size1, size2 };
	mat.create(3, sizes);
}

static float calcSum(const cv::Mat1f& src, int srcu, int srcv, int w, int h)
{
	float sum = 0;
	for (int dv = 0; dv < h; dv++)
		for (int du = 0; du < w; du++)
			sum += src(srcv + dv, srcu + du);;
	return sum;
}

static float calcMean(const cv::Mat1f& src, int srcu, int srcv, int w, int h)
{
	float sum = 0;
	int cnt = 0;
	for (int dv = 0; dv < h; dv++)
	{
		for (int du = 0; du < w; du++)
		{
			const float d = src(srcv + dv, srcu + du);
			if (d >= 0)
			{
				sum += d;
				cnt++;
			}

		}
	}
	return sum / std::max(cnt, 1);
}

static void reduceTranspose(const cv::Mat1f& src, cv::Mat1f& dst, int stixelW, int stixelH,
	bool hasInvalidValue = false)
{
	const int umax = src.cols / stixelW;
	const int vmax = src.rows / stixelH;

	dst.create(umax, vmax);

	if (hasInvalidValue)
	{
		for (int dstv = 0, srcv = 0; dstv < vmax; dstv++, srcv += stixelH)
			for (int dstu = 0, srcu = 0; dstu < umax; dstu++, srcu += stixelW)
				dst(dstu, dstv) = calcMean(src, srcu, srcv, stixelW, stixelH);
	}
	else
	{
		const float invArea = 1.f / (stixelW * stixelH);
		for (int dstv = 0, srcv = 0; dstv < vmax; dstv++, srcv += stixelH)
			for (int dstu = 0, srcu = 0; dstu < umax; dstu++, srcu += stixelW)
				dst(dstu, dstv) = invArea * calcSum(src, srcu, srcv, stixelW, stixelH);
	}
}

static Line calcRoadModelCamera(const CameraParameters& camera)
{
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);
	const float a = (camera.baseline / camera.height) * cosTilt;
	const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
	return Line(a, b);
}

static cv::Vec2d calcCostScale(const cv::Mat1f& predict)
{
	const int chns = predict.size[0];
	std::vector<double> minvs(chns);
	std::vector<double> maxvs(chns);

	OMP_PARALLEL_FOR
	for (int ch = 0; ch < chns; ch++)
		cv::minMaxIdx(getch(predict, ch), &minvs[ch], &maxvs[ch]);

	const double minv = *std::min_element(std::begin(minvs), std::end(minvs));
	const double maxv = *std::max_element(std::begin(maxvs), std::end(maxvs));

	const double a = -255. / (maxv - minv);
	const double b = -a * maxv;
	return cv::Vec2d(a, b);
}

static void calcSAT(const cv::Mat1f& src, cv::Mat1f& dst, int ch, const cv::Vec2d& scale)
{
	const cv::Mat1f channel = getch(src, ch);
	const int umax = src.size[1];
	const int vmax = src.size[2];

	const float a = static_cast<float>(scale[0]);
	const float b = static_cast<float>(scale[1]);

	for (int u = 0; u < umax; u++)
	{
		const float* ptrSrc = channel.ptr<float>(u);
		float* ptrDst = dst.ptr<float>(u, ch);
		float tmpSum = 0.f;
		for (int v = 0; v < vmax; v++)
		{
			tmpSum += a * ptrSrc[v] + b;
			ptrDst[v] = tmpSum;
		}
	}
}

static inline std::pair<Line, float> LSFitGrd(float sxx, float sxy, float syy, float sx, float sy,
	float sw, float ma, float mb)
{
	constexpr float wa = sigmaDSq[G] / sigmaSq_aG;
	constexpr float wb = sigmaDSq[G] / sigmaSq_bG;

	// solve below linear equation
	// | sxx + wa : sx      ||a| = | sxy + wa x ua |
	// | sx       : sw + wb ||b|   | sy  + wb x ub |

	// apply prior
	sxx += wa;
	sw  += wb;
	sxy += wa * ma;
	sy  += wb * mb;

	const float det = sxx * sw - sx * sx;
	if (det < std::numeric_limits<float>::epsilon())
		return { Line(), Cdef };

	// compute solution
	const float invdet = 1 / det;
	const float a = invdet * (sw * sxy - sx * sy);
	const float b = invdet * (sxx * sy - sx * sxy);

	// compute fitting error
	const float A = sxx * a * a + 2 * sx * a * b + sw * b * b;
	const float B = -2 * (sxy * a + sy * b);
	const float C = syy + wa * ma * ma + wb * mb * mb;

	return { Line(a, b), A + B + C };
}

static inline std::pair<Line, float> LSFitObj(float syy, float sy, float sw)
{
	if (sw < std::numeric_limits<float>::epsilon())
		return { Line(), Cdef };

	// compute solution
	const float b = sy / sw;

	// compute fitting error
	const float A = sw * b * b;
	const float B = -2 * sy * b;
	const float C = syy;

	return { Line(0, b), A + B + C };
}

static void calcDisparitySigmaGrd(cv::Mat1f& invSigmaGSq, int vmax, const CameraParameters& camera,
	const Line& road)
{
	invSigmaGSq.create(1, vmax);

	const float bf = camera.baseline * camera.fu;
	const float invHcam = 1.f / camera.height;

	for (int v = 0; v < vmax; v++)
	{
		const float fn = std::max(road(v), 0.f);
		const float sigmaRSq = squared(invHcam * fn) * sigmaHSq + squared(invHcam * bf) * sigmaASq;
		const float sigmaGSq = sigmaDSq[G] + sigmaRSq;
		invSigmaGSq(v) = 1.f / sigmaGSq;
	}
}

static void calcDisparitySigmaObj(cv::Mat1f& invSigmaOSq, int dmax, const CameraParameters& camera)
{
	invSigmaOSq.create(1, dmax);

	const float bf = camera.baseline * camera.fu;
	const float invDeltaD = deltaZ / bf;

	for (int fn = 0; fn < dmax; fn++)
	{
		const float sigmaZSq = squared(invDeltaD * fn * fn);
		const float sigmaOSq = sigmaDSq[O] + sigmaZSq;
		invSigmaOSq(fn) = 1.f / sigmaOSq;
	}
}

static inline float priorCostGG(float dGrdB, float dGrdT)
{
	const float delta = dGrdB - dGrdT;
	if (delta > 0)
		return alphaGapP + betaGapP * delta;
	if (delta < 0)
		return alphaGapN - betaGapN * delta;
	return 0.f;
}

static inline float priorCostGO(float dGrdB, float dObjT)
{
	const float delta = dGrdB - dObjT;
	if (delta > 0)
		return alphaGravP + betaGravP * delta;
	if (delta < 0)
		return alphaGravN - betaGravN * delta;
	return 0.f;
}

static inline float priorCostOG(float dObjB, float dGrdT)
{
	const float delta = dObjB - dGrdT;
	if (delta < 0)
		return Cinf;
	return 0.f;
}

static inline float priorCostOO(float dObjB, float dObjT)
{
	const float delta = dObjT - dObjB;
	if (delta > 0)
		return alphaOrd + betaOrd * delta;
	return 0.f;
}

static inline short packIndex(int geoId, int v)
{
	return (v << GEO_ID_BIT) | geoId;
}

static inline cv::Point unpackIndex(short packed)
{
	return { packed & ((1 << GEO_ID_BIT) - 1), packed >> GEO_ID_BIT };
}

struct BestCost
{
	inline void init(const cv::Vec3f& _costs, float _dispO)
	{
		costs = _costs;

		points[G] = packIndex(G, 0);
		points[O] = packIndex(O, 0);
		points[S] = packIndex(S, 0);

		dispG = { 0, 0 };
		dispO = _dispO;
	}

	inline void init(const cv::Vec3f& _costs, const cv::Vec3b& _labels, float _dispO)
	{
		costs = _costs;
		labels = _labels;

		points[G] = packIndex(G, 0);
		points[O] = packIndex(O, 0);
		points[S] = packIndex(S, 0);

		dispG = { 0, 0 };
		dispO = _dispO;
	}

	template <int C1, int C2>
	inline void update(int vT, float cost)
	{
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, const Line& line)
	{
		static_assert(C1 == G, "C1 must be class Grd");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			dispG = { line.a, line.b };
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, float disp)
	{
		static_assert(C1 == O, "C1 must be class Obj");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			dispO = disp;
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, int label)
	{
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			labels[C1] = label;
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, const Line& line, int label)
	{
		static_assert(C1 == G, "C1 must be class Grd");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			labels[C1] = label;
			dispG = { line.a, line.b };
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, float disp, int label)
	{
		static_assert(C1 == O, "C1 must be class Obj");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			labels[C1] = label;
			dispO = disp;
		}
	}

	cv::Vec3f costs;
	cv::Vec3s points;
	cv::Vec3b labels;
	cv::Vec2f dispG;
	float dispO;
};

static void processOneColumn(int u, const cv::Mat1f& disparity, const cv::Mat1f& confidence,
	const Line& road, const cv::Mat1f& invSigmaGSq, const cv::Mat1f& invSigmaOSq,
	cv::Mat3f& costTable, cv::Mat3s& indexTable, cv::Mat2f& dispTableG, cv::Mat1f& dispTableO)
{
	const int vmax = disparity.cols;
	const int dmax = invSigmaOSq.cols;
	const int vhor = static_cast<int>(road.vhor());

	// compute Summed Area Tables (SAT)
	const float* disparityU = disparity.ptr<float>(u);
	const float* confidenceU = confidence.ptr<float>(u);
	PlaneSAT SAT(vmax);
	for (int v = 0; v < vmax; v++)
		SAT.add(v, disparityU[v], confidenceU[v]);

	////////////////////////////////////////////////////////////////////////////////////////////
	// compute cost tables
	//
	// for paformance optimization, loop is split at vhor and unnecessary computation is ommited
	////////////////////////////////////////////////////////////////////////////////////////////
	cv::Vec3f* costTableU = costTable.ptr<cv::Vec3f>(u);
	cv::Vec3s* indexTableU = indexTable.ptr<cv::Vec3s>(u);
	cv::Vec2f* dispTableGU = dispTableG.ptr<cv::Vec2f>(u);
	float* dispTableOU = dispTableO.ptr<float>(u);

	// process vB = 0 to vhor
	// in this range, the class ground is not evaluated
	for (int vB = 0; vB <= vhor; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const float errorS = SAT.syy;
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = invSigmaOSq(d) * errorO;
			const float dataCostS = invSigmaDSq[S] * errorS;

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, lineO.b);
		}

		for (int vT = 1; vT <= vB; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const float errorS = SAT.syy;
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// compute data cost
			const float dataCostO = invSigmaOSq(d) * errorO;
			const float dataCostS = invSigmaDSq[S] * errorS;

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispO = dispTableOU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);
			const float costOS = dataCostO + prevCost[S] + Cmc;
			const float costSO = dataCostS + prevCost[O] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, lineO.b);
			bestCost.update<O, S>(vT, costOS, lineO.b);
			bestCost.update<S, O>(vT, costSO);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		dispTableGU[vB] = bestCost.dispG;
		dispTableOU[vB] = bestCost.dispO;
	}

	// process vhor + 1 to vmax
	// in this range, the class sky is not evaluated
	for (int vB = vhor + 1; vB < vmax; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = invSigmaOSq(d) * errorO;
			const float dataCostS = Cinf;

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, lineO.b);
		}

		// process vT = 1 to vhor
		// in this range, transition from/to ground is not allowed
		for (int vT = 1; vT <= vhor; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// compute data cost
			const float dataCostO = invSigmaOSq(d) * errorO;

			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispO = dispTableOU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);
			const float costOS = dataCostO + prevCost[S] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, lineO.b);
			bestCost.update<O, S>(vT, costOS, lineO.b);
		}

		// process vT = vhor + 1 to vB
		// in this range, transition from sky is not allowed
		for (int vT = vhor + 1; vT <= vB; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineG, errorG] = LSFitGrd(SAT.sxx, SAT.sxy, SAT.syy, SAT.sx, SAT.sy, SAT.sw, road.a, road.b);
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// compute data cost
			const float dataCostG = invSigmaGSq(vB) * errorG;
			const float dataCostO = invSigmaOSq(d) * errorO;

			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispG = Line(dispTableGU[vT - 1])(vT - 1);
			const float prevDispO = dispTableOU[vT - 1];

			// compute total cost
			const float costGG = dataCostG + prevCost[G] + Cmc + priorCostGG(lineG(vT), prevDispG);
			const float costGO = dataCostG + prevCost[O] + Cmc + priorCostGO(lineG(vT), prevDispO);
			const float costOG = dataCostO + prevCost[G] + Cmc + priorCostOG(lineO.b, prevDispG);
			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);

			// update best cost
			bestCost.update<G, G>(vT, costGG, lineG);
			bestCost.update<G, O>(vT, costGO, lineG);
			bestCost.update<O, G>(vT, costOG, lineO.b);
			bestCost.update<O, O>(vT, costOO, lineO.b);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		dispTableGU[vB] = bestCost.dispG;
		dispTableOU[vB] = bestCost.dispO;
	}
}

std::pair<float, int> calcMinCostAndLabel(const cv::Mat1f& SATsem,
	const std::vector<int>& labels, int vB, int vT = 0)
{
	float minCost = Cinf;
	int minLabel = -1;
	for (int label : labels)
	{
		const float cost = vT > 0 ? SATsem(label, vB) - SATsem(label, vT - 1) : SATsem(label, vB);
		if (cost < minCost)
		{
			minCost = cost;
			minLabel = label;
		}
	}
	return { minCost, minLabel };
}

static void processOneColumn(int u, const cv::Mat1f& disparity, const cv::Mat1f& confidence,
	const Line& road, const cv::Mat1f& invSigmaGSq, const cv::Mat1f& invSigmaOSq,
	const cv::Mat1f& SATsem, const std::vector<int> G2L[], cv::Mat3f& costTable,
	cv::Mat3s& indexTable, cv::Mat3b& labelTable, cv::Mat2f& dispTableG, cv::Mat1f& dispTableO)
{
	const int vmax = disparity.cols;
	const int dmax = invSigmaOSq.cols;
	const int vhor = static_cast<int>(road.vhor());
	const int chns = SATsem.size[1];

	// compute Summed Area Tables (SAT) for slanted plane
	const float* disparityU = disparity.ptr<float>(u);
	const float* confidenceU = confidence.ptr<float>(u);
	PlaneSAT SAT(vmax);
	for (int v = 0; v < vmax; v++)
		SAT.add(v, disparityU[v], confidenceU[v]);

	////////////////////////////////////////////////////////////////////////////////////////////
	// compute cost tables
	//
	// for paformance optimization, loop is split at vhor and unnecessary computation is ommited
	////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat1f SATsemU = getch(SATsem, u);
	cv::Vec3f* costTableU = costTable.ptr<cv::Vec3f>(u);
	cv::Vec3s* indexTableU = indexTable.ptr<cv::Vec3s>(u);
	cv::Vec3b* labelTableU = labelTable.ptr<cv::Vec3b>(u);
	cv::Vec2f* dispTableGU = dispTableG.ptr<cv::Vec2f>(u);
	float* dispTableOU = dispTableO.ptr<float>(u);

	// process vB = 0 to vhor
	// in this range, the class ground is not evaluated
	for (int vB = 0; vB <= vhor; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const float errorS = SAT.syy;
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB);
			const auto [minSemCostS, minLabelS] = calcMinCostAndLabel(SATsemU, G2L[S], vB);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = invSigmaOSq(d) * errorO + wsem * minSemCostO;
			const float dataCostS = invSigmaDSq[S] * errorS + wsem * minSemCostS;

			// initialize best cost
			bestCost.init(cv::Vec3f(dataCostG, dataCostO, dataCostS),
				cv::Vec3b(0, minLabelO, minLabelS), lineO.b);
		}

		for (int vT = 1; vT <= vB; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const float errorS = SAT.syy;
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT);
			const auto [minSemCostS, minLabelS] = calcMinCostAndLabel(SATsemU, G2L[S], vB, vT);

			// compute data cost
			const float dataCostO = invSigmaOSq(d) * errorO + wsem * minSemCostO;
			const float dataCostS = invSigmaDSq[S] * errorS + wsem * minSemCostS;

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispO = dispTableOU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);
			const float costOS = dataCostO + prevCost[S] + Cmc;
			const float costSO = dataCostS + prevCost[O] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, lineO.b, minLabelO);
			bestCost.update<O, S>(vT, costOS, lineO.b, minLabelO);
			bestCost.update<S, O>(vT, costSO, minLabelS);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		labelTableU[vB] = bestCost.labels;
		dispTableGU[vB] = bestCost.dispG;
		dispTableOU[vB] = bestCost.dispO;
	}

	// process vhor + 1 to vmax
	// in this range, the class sky is not evaluated
	for (int vB = vhor + 1; vB < vmax; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = invSigmaOSq(d) * errorO + wsem * minSemCostO;
			const float dataCostS = Cinf;

			// initialize best cost
			bestCost.init(cv::Vec3f(dataCostG, dataCostO, dataCostS),
				cv::Vec3b(0, minLabelO, 0), lineO.b);
		}

		// process vT = 1 to vhor
		// in this range, transition from/to ground is not allowed
		for (int vT = 1; vT <= vhor; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT);

			// compute data cost
			const float dataCostO = invSigmaOSq(d) * errorO + wsem * minSemCostO;

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispO = dispTableOU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);
			const float costOS = dataCostO + prevCost[S] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, lineO.b, minLabelO);
			bestCost.update<O, S>(vT, costOS, lineO.b, minLabelO);
		}

		// process vT = vhor + 1 to vB
		// in this range, transition from sky is not allowed
		for (int vT = vhor + 1; vT <= vB; vT++)
		{
			// compute sums within the range of vB to vT
			SAT.setInterval(vB, vT);

			// least squares fit and compute fit error
			const auto [lineG, errorG] = LSFitGrd(SAT.sxx, SAT.sxy, SAT.syy, SAT.sx, SAT.sy, SAT.sw, road.a, road.b);
			const auto [lineO, errorO] = LSFitObj(SAT.syy, SAT.sy, SAT.sw);
			const int d = std::min(cvRound(lineO.b), dmax - 1);

			// minimization over the semantic labels
			const auto [minSemCostG, minLabelG] = calcMinCostAndLabel(SATsemU, G2L[G], vB, vT);
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT);

			// compute data cost
			const float dataCostG = invSigmaGSq(vB) * errorG + wsem * minSemCostG;
			const float dataCostO = invSigmaOSq(d) * errorO + wsem * minSemCostO;

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];
			const float prevDispG = Line(dispTableGU[vT - 1])(vT - 1);
			const float prevDispO = dispTableOU[vT - 1];

			const float costGG = dataCostG + prevCost[G] + Cmc + priorCostGG(lineG(vT), prevDispG);
			const float costGO = dataCostG + prevCost[O] + Cmc + priorCostGO(lineG(vT), prevDispO);
			const float costOG = dataCostO + prevCost[G] + Cmc + priorCostOG(lineO.b, prevDispG);
			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(lineO.b, prevDispO);

			// update best cost
			bestCost.update<G, G>(vT, costGG, lineG, minLabelG);
			bestCost.update<G, O>(vT, costGO, lineG, minLabelG);
			bestCost.update<O, G>(vT, costOG, lineO.b, minLabelO);
			bestCost.update<O, O>(vT, costOO, lineO.b, minLabelO);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		labelTableU[vB] = bestCost.labels;
		dispTableGU[vB] = bestCost.dispG;
		dispTableOU[vB] = bestCost.dispO;
	}
}

static void extractStixels(const cv::Mat3f& costTable, const cv::Mat3s& indexTable,
	const cv::Mat2f& dispTableG, const cv::Mat1f& dispTableO, std::vector<Stixel>& stixels)
{
	const int umax = costTable.rows;
	const int vmax = costTable.cols;

	for (int u = 0; u < umax; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, vmax - 1)[c];
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, vmax - 1);
			}
		}

		while (minPos.y > 0)
		{
			const int geoId = minPos.x;
			const int v = minPos.y;

			const cv::Point p1 = minPos;
			const cv::Point p2 = unpackIndex(indexTable(u, v)[geoId]);

			Stixel stixel;
			stixel.uL = u;
			stixel.vT = p2.y + 1;
			stixel.vB = p1.y + 1;
			stixel.width = 1;
			stixel.geoId = geoId;
			stixel.semId = 0;

			if (geoId == G)
				stixel.disp = dispTableG(u, v);
			if (geoId == O)
				stixel.disp = cv::Vec2f(0, dispTableO(u, v));
			if (geoId == S)
				stixel.disp = cv::Vec2f(0, 0);

			stixels.push_back(stixel);

			minPos = p2;
		}
	}
}

static void extractStixels(const cv::Mat3f& costTable, const cv::Mat3s& indexTable, const cv::Mat3b& labelTable,
	const cv::Mat2f& dispTableG, const cv::Mat1f& dispTableO, std::vector<Stixel>& stixels)
{
	const int umax = costTable.rows;
	const int vmax = costTable.cols;

	for (int u = 0; u < umax; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, vmax - 1)[c];
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, vmax - 1);
			}
		}

		while (minPos.y > 0)
		{
			const int geoId = minPos.x;
			const int v = minPos.y;

			const cv::Point p1 = minPos;
			const cv::Point p2 = unpackIndex(indexTable(u, v)[geoId]);

			Stixel stixel;
			stixel.uL = u;
			stixel.vT = p2.y + 1;
			stixel.vB = p1.y + 1;
			stixel.width = 1;
			stixel.geoId = geoId;
			stixel.semId = labelTable(u, v)[geoId];

			if (geoId == G)
				stixel.disp = dispTableG(u, v);
			if (geoId == O)
				stixel.disp = cv::Vec2f(0, dispTableO(u, v));
			if (geoId == S)
				stixel.disp = cv::Vec2f(0, 0);

			stixels.push_back(stixel);

			minPos = p2;
		}
	}
}

class SlantedStixelsImpl : public SlantedStixels
{
public:

	SlantedStixelsImpl(const Parameters& param) : param_(param)
	{
		init();
	}

	void init()
	{
		const auto& L2G = param_.geometry;
		//CV_Assert(!L2G.empty());

		const int chns = static_cast<int>(L2G.size());
		for (int ch = 0; ch < chns; ch++)
		{
			const int geoId = L2G[ch];
			if (geoId >= 0 && geoId <= 2)
				G2L_[geoId].push_back(ch);
		}
	}

	void compute(const cv::Mat& disparity, const cv::Mat& confidence,
		std::vector<Stixel>& stixels) override
	{
		stixels.clear();

		const int stixeW = param_.stixelWidth;
		const int stixeH = param_.stixelYResolution;

		const CameraParameters& camera = param_.camera;

		CV_Assert(disparity.type() == CV_32F && confidence.type() == CV_32F);
		CV_Assert(disparity.size() == confidence.size());
		CV_Assert(stixeW == STIXEL_WIDTH_4 || stixeW == STIXEL_WIDTH_8);
		CV_Assert(stixeH == STIXEL_Y_RESOLUTION_4 || stixeH == STIXEL_Y_RESOLUTION_8);

		//////////////////////////////////////////////////////////////////////////////
		// process depth input
		//////////////////////////////////////////////////////////////////////////////

		// reduce and reorder disparity map
		reduceTranspose(disparity, disparity_, stixeW, stixeH, true);
		reduceTranspose(confidence, confidence_, stixeW, stixeH);

		// estimate road model from camera tilt and height
		Line road = calcRoadModelCamera(camera);
		road.a *= stixeH; // correct slope according to stixe Y resolution

		//////////////////////////////////////////////////////////////////////////////
		// dynamic programming
		//////////////////////////////////////////////////////////////////////////////

		const int umax = disparity_.rows;
		const int vmax = disparity_.cols;
		const int dmax = param_.dmax;

		costTable_.create(umax, vmax);
		indexTable_.create(umax, vmax);
		dispTableG_.create(umax, vmax);
		dispTableO_.create(umax, vmax);

		calcDisparitySigmaGrd(invSigmaGSq_, vmax, camera, road);
		calcDisparitySigmaObj(invSigmaOSq_, dmax, camera);

		OMP_PARALLEL_FOR
		for (int u = 0; u < umax; u++)
		{
			processOneColumn(u, disparity_, confidence_, road, invSigmaGSq_, invSigmaOSq_,
				costTable_, indexTable_, dispTableG_, dispTableO_);
		}

		extractStixels(costTable_, indexTable_, dispTableG_, dispTableO_, stixels);

		for (auto& stixel : stixels)
		{
			stixel.uL *= stixeW;
			stixel.vT *= stixeH;
			stixel.vB *= stixeH;
			stixel.width = stixeW;
			if (stixel.geoId == G)
				stixel.disp[0] /= stixeH;
		}
	}

	void compute(const cv::Mat& disparity, const cv::Mat& confidence, const cv::Mat& predict,
		std::vector<Stixel>& stixels) override
	{
		stixels.clear();

		const int stixeW = param_.stixelWidth;
		const int stixeH = param_.stixelYResolution;

		const CameraParameters& camera = param_.camera;

		CV_Assert(disparity.type() == CV_32F && confidence.type() == CV_32F && predict.type() == CV_32F);
		CV_Assert(disparity.size() == confidence.size());
		CV_Assert(disparity.rows == predict.size[1] && disparity.cols == predict.size[2]);
		CV_Assert(stixeW == STIXEL_WIDTH_4 || stixeW == STIXEL_WIDTH_8);
		CV_Assert(stixeH == STIXEL_Y_RESOLUTION_4 || stixeH == STIXEL_Y_RESOLUTION_8);

		//////////////////////////////////////////////////////////////////////////////
		// process depth input
		//////////////////////////////////////////////////////////////////////////////

		// reduce and reorder disparity map
		reduceTranspose(disparity, disparity_, stixeW, stixeH, true);
		reduceTranspose(confidence, confidence_, stixeW, stixeH);

		// estimate road model from camera tilt and height
		Line road = calcRoadModelCamera(param_.camera);
		road.a *= stixeH; // correct slope according to stixe Y resolution

		//////////////////////////////////////////////////////////////////////////////
		// process semantic input
		//////////////////////////////////////////////////////////////////////////////

		const int chns = predict.size[0];
		const int umax = disparity_.rows;
		const int vmax = disparity_.cols;
		const int dmax = param_.dmax;

		create3d(predict_, chns, umax, vmax);
		create3d(SATsem_, umax, chns, vmax);

		OMP_PARALLEL_FOR
		for (int ch = 0; ch < chns; ch++)
			reduceTranspose(getch(predict, ch), getch(predict_, ch), stixeW, stixeH);

		const auto costScale = calcCostScale(predict_);

		OMP_PARALLEL_FOR
		for (int ch = 0; ch < chns; ch++)
			calcSAT(predict_, SATsem_, ch, costScale);

		//////////////////////////////////////////////////////////////////////////////
		// dynamic programming
		//////////////////////////////////////////////////////////////////////////////
		
		costTable_.create(umax, vmax);
		indexTable_.create(umax, vmax);
		labelTable_.create(umax, vmax);
		dispTableG_.create(umax, vmax);
		dispTableO_.create(umax, vmax);

		calcDisparitySigmaGrd(invSigmaGSq_, vmax, camera, road);
		calcDisparitySigmaObj(invSigmaOSq_, dmax, camera);

		OMP_PARALLEL_FOR
		for (int u = 0; u < umax; u++)
		{
			processOneColumn(u, disparity_, confidence_, road, invSigmaGSq_, invSigmaOSq_,
				SATsem_, G2L_, costTable_, indexTable_, labelTable_, dispTableG_, dispTableO_);
		}

		extractStixels(costTable_, indexTable_, labelTable_, dispTableG_, dispTableO_, stixels);

		for (auto& stixel : stixels)
		{
			stixel.uL *= stixeW;
			stixel.vT *= stixeH;
			stixel.vB *= stixeH;
			stixel.width = stixeW;
			if (stixel.geoId == G)
				stixel.disp[0] /= stixeH;
		}
	}

	void setParameters(const Parameters& param) override
	{
		param_ = param;
	}

private:

	cv::Mat1f disparity_;
	cv::Mat1f confidence_;
	cv::Mat3f costTable_;
	cv::Mat3s indexTable_;
	cv::Mat3b labelTable_;
	cv::Mat2f dispTableG_;
	cv::Mat1f dispTableO_;

	cv::Mat1f invSigmaGSq_;
	cv::Mat1f invSigmaOSq_;

	cv::Mat1f predict_;
	cv::Mat1f SATsem_;
	std::vector<int> G2L_[3];

	Parameters param_;
};

cv::Ptr<SlantedStixels> SlantedStixels::create(const Parameters& param)
{
	return cv::makePtr<SlantedStixelsImpl>(param);
}
