

class CV_EXPORTS_W MyCascadeClassifier
{
public:
    CV_WRAP MyCascadeClassifier();
    CV_WRAP MyCascadeClassifier( const string& filename );
    virtual ~MyCascadeClassifier();
    
    CV_WRAP virtual bool empty() const;
    CV_WRAP bool load( const string& filename );
    virtual bool read( const FileNode& node );
    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                          CV_OUT vector<Rect>& objects,
                                          double scaleFactor=1.1,
                                          int minNeighbors=3, int flags=0,
                                          Size minSize=Size(),
                                          Size maxSize=Size() );
    
    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                          CV_OUT vector<Rect>& objects,
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor=1.1,
                                          int minNeighbors=3, int flags=0,
                                          Size minSize=Size(),
                                          Size maxSize=Size(),
                                          bool outputRejectLevels=false );
    
    
    bool isOldFormatCascade() const;
    virtual Size getOriginalWindowSize() const;
    int getFeatureType() const;
    bool setImage( const Mat& );
    
protected:
    //virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
    //                                int stripSize, int yStep, double factor, vector<Rect>& candidates );
    
    virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                   int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                   vector<int>& rejectLevels, vector<double>& levelWeights, bool outputRejectLevels=false);
    
protected:
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
        FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8 };
    
    friend class CascadeClassifierInvoker;
    
    template<class FEval>
    friend int predictOrdered( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);
    
    template<class FEval>
    friend int predictCategorical( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);
    
    template<class FEval>
    friend int predictOrderedStump( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);
    
    template<class FEval>
    friend int predictCategoricalStump( MyCascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);
    
    bool setImage( Ptr<FeatureEvaluator>& feval, const Mat& image);
    virtual int runAt( Ptr<FeatureEvaluator>& feval, Point pt, double& weight );
    
    class Data
    {
    public:
        struct CV_EXPORTS DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };
        
        struct CV_EXPORTS DTree
        {
            int nodeCount;
        };
        
        struct CV_EXPORTS Stage
        {
            int first;
            int ntrees;
            float threshold;
        };
        
        bool read(const FileNode &node);
        
        bool isStumpBased;
        
        int stageType;
        int featureType;
        int ncategories;
        Size origWinSize;
        
        vector<Stage> stages;
        vector<DTree> classifiers;
        vector<DTreeNode> nodes;
        vector<float> leaves;
        vector<int> subsets;
    };
    
    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;
    
public:
    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual cv::Mat generateMask(const cv::Mat& src)=0;
        virtual void initializeMask(const cv::Mat& /*src*/) {};
    };
    void setMaskGenerator(Ptr<MaskGenerator> maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();
    
    void setFaceDetectionMaskGenerator();
    
protected:
    Ptr<MaskGenerator> maskGenerator;
}   ;
