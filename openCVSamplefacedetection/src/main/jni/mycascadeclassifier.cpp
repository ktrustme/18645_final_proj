#include "precomp.hpp"
#include <cstdio>
#include "cascadedetect.hpp"
#include "objdetect.hpp"


#include <string>

using namespace cv;
using namespace std;

#include "mycascadeclassifier.hpp"
//---------------------------------------- Classifier Cascade --------------------------------------------

MyCascadeClassifier::MyCascadeClassifier()
{
}

MyCascadeClassifier::MyCascadeClassifier(const string& filename)
{
    load(filename);
}

MyCascadeClassifier::~MyCascadeClassifier()
{
}

bool MyCascadeClassifier::empty() const
{
    return oldCascade.empty() && data.stages.empty();
}

bool MyCascadeClassifier::load(const string& filename)
{
    oldCascade.release();
    data = Data();
    featureEvaluator.release();
    
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    
    if( read(fs.getFirstTopLevelNode()) )
        return true;
    
    fs.release();
    
    oldCascade = Ptr<CvHaarClassifierCascade>((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
    return !oldCascade.empty();
}

int MyCascadeClassifier::runAt( Ptr<FeatureEvaluator>& evaluator, Point pt, double& weight )
{
    CV_Assert( oldCascade.empty() );
    
    assert( data.featureType == FeatureEvaluator::HAAR ||
           data.featureType == FeatureEvaluator::LBP ||
           data.featureType == FeatureEvaluator::HOG );
    
    if( !evaluator->setWindow(pt) )
        return -1;
    if( data.isStumpBased )
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrderedStump<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategoricalStump<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrderedStump<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
    else
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrdered<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategorical<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrdered<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
}

bool MyCascadeClassifier::setImage( Ptr<FeatureEvaluator>& evaluator, const Mat& image )
{
    return empty() ? false : evaluator->setImage(image, data.origWinSize);
}

void MyCascadeClassifier::setMaskGenerator(Ptr<MaskGenerator> _maskGenerator)
{
    maskGenerator=_maskGenerator;
}
Ptr<MyCascadeClassifier::MaskGenerator> MyCascadeClassifier::getMaskGenerator()
{
    return maskGenerator;
}

void MyCascadeClassifier::setFaceDetectionMaskGenerator()
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    setMaskGenerator(tegra::getMyCascadeClassifierMaskGenerator(*this));
#else
    setMaskGenerator(Ptr<MyCascadeClassifier::MaskGenerator>());
#endif
}

class CascadeClassifierInvoker : public ParallelLoopBody
{
public:
    CascadeClassifierInvoker( MyCascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor,
                             vector<Rect>& _vec, vector<int>& _levels, vector<double>& _weights, bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        processingRectSize = _sz1;
        stripSize = _stripSize;
        yStep = _yStep;
        scalingFactor = _factor;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
    }
    
    void operator()(const Range& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
        
        Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));
        
        int y1 = range.start * stripSize;
        int y2 = min(range.end * stripSize, processingRectSize.height);
        for( int y = y1; y < y2; y += yStep )
        {
            for( int x = 0; x < processingRectSize.width; x += yStep )
            {
                if ( (!mask.empty()) && (mask.at<uchar>(Point(x,y))==0)) {
                    continue;
                }
                
                double gypWeight;
                int result = classifier->runAt(evaluator, Point(x, y), gypWeight);
                
#if defined (LOG_CASCADE_STATISTIC)
                
                logger.setPoint(Point(x, y), result);
#endif
                if( rejectLevels )
                {
                    if( result == 1 )
                        result =  -(int)classifier->data.stages.size();
                    if( classifier->data.stages.size() + result < 4 )
                    {
                        mtx->lock();
                        rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
                        rejectLevels->push_back(-result);
                        levelWeights->push_back(gypWeight);
                        mtx->unlock();
                    }
                }
                else if( result > 0 )
                {
                    mtx->lock();
                    rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),
                                               winSize.width, winSize.height));
                    mtx->unlock();
                }
                if( result == 0 )
                    x += yStep;
            }
        }
    }
    
    MyCascadeClassifier* classifier;
    vector<Rect>* rectangles;
    Size processingRectSize;
    int stripSize, yStep;
    double scalingFactor;
    vector<int> *rejectLevels;
    vector<double> *levelWeights;
    Mat mask;
    Mutex* mtx;
};

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };


bool MyCascadeClassifier::detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                          int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                          vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
{
    if( !featureEvaluator->setImage( image, data.origWinSize ) )
        return false;
    
#if defined (LOG_CASCADE_STATISTIC)
    logger.setImage(image);
#endif
    
    Mat currentMask;
    if (!maskGenerator.empty()) {
        currentMask=maskGenerator->generateMask(image);
    }
    
    vector<Rect> candidatesVector;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    Mutex mtx;
    if( outputRejectLevels )
    {
        parallel_for_(Range(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
                                                                     candidatesVector, rejectLevels, levelWeights, true, currentMask, &mtx));
        levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
        weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
    }
    else
    {
        parallel_for_(Range(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
                                                                     candidatesVector, rejectLevels, levelWeights, false, currentMask, &mtx));
    }
    candidates.insert( candidates.end(), candidatesVector.begin(), candidatesVector.end() );
    
#if defined (LOG_CASCADE_STATISTIC)
    logger.write();
#endif
    
    return true;
}

bool MyCascadeClassifier::isOldFormatCascade() const
{
    return !oldCascade.empty();
}


int MyCascadeClassifier::getFeatureType() const
{
    return featureEvaluator->getFeatureType();
}

Size MyCascadeClassifier::getOriginalWindowSize() const
{
    return data.origWinSize;
}

bool MyCascadeClassifier::setImage(const Mat& image)
{
    return featureEvaluator->setImage(image, data.origWinSize);
}



//That's the function we are going to optimize!
void MyCascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                         vector<int>& rejectLevels,
                                         vector<double>& levelWeights,
                                         double scaleFactor, int minNeighbors,
                                         int flags, Size minObjectSize, Size maxObjectSize,
                                         bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;
    
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );
    
    if( empty() )
        return;
    
    if( isOldFormatCascade() )
    {
        MemStorage storage(cvCreateMemStorage(0));
        CvMat _image = image;
        CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                                    minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        objects.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
        return;
    }
    
    objects.clear();
    
    if (!maskGenerator.empty()) {
        maskGenerator->initializeMask(image);
    }
    
    
    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();
    
    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }
    
    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
    vector<Rect> candidates;
    
    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();
        
        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
        Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );
        
        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;
        
        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );
        
        int yStep;
        if( getFeatureType() == cv::FeatureEvaluator::HOG )
        {
            yStep = 4;
        }
        else
        {
            yStep = factor > 2. ? 1 : 2;
        }
        
        int stripCount, stripSize;
        
        const int PTS_PER_THREAD = 1000;
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;
        
        if( !detectSingleScale( scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
                               rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }
    
    
    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());
    
    if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, minNeighbors, GROUP_EPS );
    }
}


void MyCascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                         double scaleFactor, int minNeighbors,
                                         int flags, Size minObjectSize, Size maxObjectSize)
{
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
                     minNeighbors, flags, minObjectSize, maxObjectSize, false );
}

bool MyCascadeClassifier::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;
    
    // load stage params
    string stageTypeStr = (string)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;
    
    string featureTypeStr = (string)root[CC_FEATURE_TYPE];
    if( featureTypeStr == CC_HAAR )
        featureType = FeatureEvaluator::HAAR;
    else if( featureTypeStr == CC_LBP )
        featureType = FeatureEvaluator::LBP;
    else if( featureTypeStr == CC_HOG )
        featureType = FeatureEvaluator::HOG;
    
    else
        return false;
    
    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );
    
    isStumpBased = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;
    
    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;
    
    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
    nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );
    
    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;
    
    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();
    
    FileNodeIterator it = fn.begin(), it_end = fn.end();
    
    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);
        
        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;
            
            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);
            
            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);
            
            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();
            
            for( ; internalNodesIter != internalNodesEnd; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    node.threshold = 0.f;
                }
                else
                {
                    node.threshold = (float)*internalNodesIter; ++internalNodesIter;
                }
                nodes.push_back(node);
            }
            
            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();
            
            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((float)*internalNodesIter);
        }
    }
    
    return true;
}

bool MyCascadeClassifier::read(const FileNode& root)
{
    if( !data.read(root) )
        return false;
    
    // load features
    featureEvaluator = FeatureEvaluator::create(data.featureType);
    FileNode fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;
    
    return featureEvaluator->read(fn);
}

template<> void Ptr<CvHaarClassifierCascade>::delete_obj()
{ cvReleaseHaarClassifierCascade(&obj); }



//----------------------------------------------  predictor functions -------------------------------------

template<class FEval>
inline int predictOrdered( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    MyCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    MyCascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    MyCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];
    
    for( int si = 0; si < nstages; si++ )
    {
        MyCascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;
        
        for( wi = 0; wi < ntrees; wi++ )
        {
            MyCascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;
            
            do
            {
                MyCascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
                double val = featureEvaluator(node.featureIdx);
                idx = val < node.threshold ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class FEval>
inline int predictCategorical( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.data.subsets[0];
    float* cascadeLeaves = &cascade.data.leaves[0];
    MyCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    MyCascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    MyCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];
    
    for(int si = 0; si < nstages; si++ )
    {
        MyCascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;
        
        for( wi = 0; wi < ntrees; wi++ )
        {
            MyCascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;
            do
            {
                MyCascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
                int c = featureEvaluator(node.featureIdx);
                const int* subset = &cascadeSubsets[(root + idx)*subsetSize];
                idx = (subset[c>>5] & (1 << (c & 31))) ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class FEval>
inline int predictOrderedStump( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    MyCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    MyCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];
    
    int nstages = (int)cascade.data.stages.size();
    for( int stageIdx = 0; stageIdx < nstages; stageIdx++ )
    {
        MyCascadeClassifier::Data::Stage& stage = cascadeStages[stageIdx];
        sum = 0.0;
        
        int ntrees = stage.ntrees;
        for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
        {
            MyCascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            double value = featureEvaluator(node.featureIdx);
            sum += cascadeLeaves[ value < node.threshold ? leafOfs : leafOfs + 1 ];
        }
        
        if( sum < stage.threshold )
            return -stageIdx;
    }
    
    return 1;
}

template<class FEval>
inline int predictCategoricalStump( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    size_t subsetSize = (cascade.data.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.data.subsets[0];
    float* cascadeLeaves = &cascade.data.leaves[0];
    MyCascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    MyCascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];
    
#ifdef HAVE_TEGRA_OPTIMIZATION
    float tmp = 0; // float accumulator -- float operations are quicker
#endif
    for( int si = 0; si < nstages; si++ )
    {
        MyCascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
#ifdef HAVE_TEGRA_OPTIMIZATION
        tmp = 0;
#else
        sum = 0;
#endif
        
        for( wi = 0; wi < ntrees; wi++ )
        {
            MyCascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            int c = featureEvaluator(node.featureIdx);
            const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
#ifdef HAVE_TEGRA_OPTIMIZATION
            tmp += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
#else
            sum += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
#endif
            nodeOfs++;
            leafOfs += 2;
        }
#ifdef HAVE_TEGRA_OPTIMIZATION
        if( tmp < stage.threshold ) {
            sum = (double)tmp;
            return -si;
        }
#else
        if( sum < stage.threshold )
            return -si;
#endif
    }
    
#ifdef HAVE_TEGRA_OPTIMIZATION
    sum = (double)tmp;
#endif
    
    return 1;
}