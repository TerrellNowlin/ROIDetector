import numpy as np
import math
import sys
import pickle
from os import listdir
from os.path import isdir, join
import matplotlib.image as mpimg
from sklearn import svm
from sklearn.cluster import KMeans as km
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.preprocessing import normalize
import multiprocessing as mp
#favorite_color = pickle.load( open( "file.pkl", "rb" ) )
def im2col(x, HF, WF):
    Xt=[]
    [h,w] = x.shape
    for j in range(int(math.floor(w/WF))):
       
       for i in range(int(math.floor(h/HF))):
            temp=x[i*HF:(i+1)*HF,j*WF:(j+1)*WF]
            temp=np.reshape(np.transpose(temp), (HF*WF,))
            Xt.append(temp)
    return np.transpose(np.asarray(Xt))

def buildNonOverlappingPatches(patchsize, img):
    [h,w,d] = img.shape
    img       = img[0:int(patchsize*math.floor(h/patchsize)), 0:int(patchsize*math.floor(w/patchsize)),:]
    Xt1      = im2col(img[:,:,0],patchsize,patchsize)
    Xt2      = im2col(img[:,:,1],patchsize,patchsize)
    Xt3      = im2col(img[:,:,2],patchsize,patchsize)
    X=np.stack((Xt1,Xt2,Xt3), axis=2)
    return [X,img]

def generateRandomSearch(pRange, kRange ,minDistRange, thresholdRange, gammaRange, nuRange, n):
    p=np.random.randint(low=pRange[0], high=pRange[1], size=n)
    k=np.random.randint(low=kRange[0], high=kRange[1], size=n)
    minDist=(minDistRange[1] - minDistRange[0]) *np.random.rand(n,1)+minDistRange[0]
    threshold=np.random.randint(low=thresholdRange[0], high=thresholdRange[1], size=n)
    gamma=(gammaRange[1] - gammaRange[0])*np.random.rand(n,1)+gammaRange[0]
    nu=(nuRange[1] - nuRange[0])*np.random.rand(n,1)
    return [p, k, minDist[:,0], threshold, gamma[:,0], nu[:,0]]



def runROIDetector(p,k,min_dist,threshold, gammae, nue):
    k=k.astype(np.int64)
    p=p.astype(np.int64)
    threshold=threshold.astype(np.int64)
    resize=.3

    filesHtrain=[]
    filesUtrain=[]
    
    filesUtest=[]
    filesHtest=[]
    
    filesUCV=[]
    filesHCV=[]


    
    unhealthyTestPatient = [f for f in listdir('testROI/unhealthy') if isdir(join('testROI/unhealthy', f))]
    healthyTestPatient  = [f for f in listdir('testROI/healthy') if isdir(join('testROI/healthy', f))]
    unhealthyTrainPatient  = [f for f in listdir('trainROI/unhealthy') if isdir(join('trainROI/unhealthy', f))]
    healthyTrainPatient  = [f for f in listdir('trainROI/healthy') if isdir(join('trainROI/healthy', f))]
    unhealthyCVPatient = [f for f in listdir('trainROI/unlabel') if isdir(join('trainROI/unlabel', f))]
    healthyCVPatient = healthyTrainPatient[0:8]
    healthyTrainPatient=healthyTrainPatient[8:len(healthyTrainPatient)-1]

    for i in healthyTrainPatient:
        dirs=listdir('trainROI/healthy/'+i)
        for j in dirs:
            if "._" not in j:
                filesHtrain.append(('trainROI/healthy/'+i+'/'+j))
            
    for i in unhealthyTrainPatient:
        dirs=listdir('trainROI/unhealthy/'+i)
        for j in dirs:
            if "._" not in j:
                filesUtrain.append(('trainROI/unhealthy/'+i+'/'+j))
                
    for i in unhealthyCVPatient:
        dirs=listdir('trainROI/unlabel/'+i)
        for j in dirs:
            if "._" not in j:
                filesUCV.append(('trainROI/unlabel/'+i+'/'+j))
    for i in healthyTestPatient:
        dirs=listdir('testROI/healthy/'+i)
        for j in dirs:
            if "._" not in j:
                filesHtest.append(('testROI/healthy/'+i+'/'+j))
            
    for i in unhealthyTestPatient:
        dirs=listdir('testROI/unhealthy/'+i)
        for j in dirs:
            if "._" not in j:
                filesUtest.append(('testROI/unhealthy/'+i+'/'+j))

    init_words=[]
    words=[] 
    final_words=[]
    healthyPatientDict={}
# Part One:----------Extract initial words from training positive set----------
    for image in (filesUtrain):
#        read image from Unhealthy class (These are the seed examples)
        img = imread(image)
        [h,w,d] = img.shape
        img = rescale(img, resize)
        [h,w,d] = img.shape
        
#        Trim the image to a bucketable size
        F    = img[0:int(p*math.floor(h/p)), 0:int(p*math.floor(w/p)),:]
        
#        Extract HOG Features
        fd = hog(F[:,:,1], orientations=8, pixels_per_cell=(p, p),cells_per_block=(1, 1))
        
#        Reorganize features so that each index matches a sample
        features=np.reshape(fd,(((F.shape[1]*F.shape[0])/(p*p)),8))
        
#        Cluster the features
        KM=km(n_clusters=k).fit(features)
        
#        Add the cluster centers to the inital dictionary
        init_words.extend(KM.cluster_centers_)
        
    init_words=np.asarray(init_words)
    occurencVec={}
### Part Two:----------------------Filter out "bad words"------------------------
    for patient in (healthyTrainPatient):
        healthyPatientDict[patient]={}
        for idx in range(len(init_words)):
            temp=healthyPatientDict[patient]
            temp[idx]=-1
            occurencVec[idx]=0
            healthyPatientDict[patient]=temp
    for patient in (healthyTrainPatient):
        imageList = listdir('trainROI/healthy/'+patient)
        HOGvec=[]
        count=0
        for image in imageList:
            if "._" not in image:
#             read image from Healthy class
                img = imread('trainROI/healthy/'+patient+'/'+image)
                [h,w,d] = img.shape
                img = rescale(img, resize, anti_aliasing=True)
                [h,w,d] = img.shape
                
#                 Trim the image to a bucketable size
                F    = img[0:int(p*math.floor(h/p)), 0:int(p*math.floor(w/p)),:]
                
#               Extract HOG Features
                fd = hog(F[:,:,1], orientations=8, pixels_per_cell=(p, p),cells_per_block=(1, 1))
                
#              Reorganize features so that each index matches a sample
                features=np.reshape(fd,(((F.shape[1]*F.shape[0])/(p*p)),8))
                if count==0:
                    HOGvec=features
                else:
                    HOGvec=np.vstack((HOGvec,features))
                count=count+1
#       For each of the words in the inital dictionary, calculate the 
#       L2-distance between the features of the current image. If the distance
#       is too small too many times, remove the word from the dictionary.
        
        initWordsIdx=0
        for rows in (init_words): 
            num_of_matches=0
            iters=0
            for n in HOGvec: 
                iters=iters+1
                r=np.linalg.norm(rows-n)
                if r < min_dist:
                    num_of_matches=num_of_matches+1
                    temp=occurencVec[initWordsIdx]
                    occurencVec[initWordsIdx]=temp+1
                iters=iters+1
            final_words.append(rows)
            temp=healthyPatientDict[patient]
            temp[initWordsIdx]=num_of_matches
            initWordsIdx=initWordsIdx+1
    
    averages=[]
    for count in range(len(healthyPatientDict[healthyTrainPatient[0]])):
        numerator=0
        for patient in healthyTrainPatient:
            temp=healthyPatientDict[patient]
            numerator=numerator+temp[count]
        average=numerator/len(healthyPatientDict)
        averages.append(average)
        idxs=np.where(np.asarray(averages)<threshold)
    idxs=idxs[0]
    featureMatrix=np.zeros(shape=(len(healthyPatientDict),len(idxs)))
    i=0    
    for patient in healthyPatientDict:
        j=0
        for idx in idxs:
            temp=healthyPatientDict[patient]
            featureMatrix[i,j]=temp[idx]
            j=j+1
        i=i+1
    validationMatrix=np.zeros(shape=(len(unhealthyCVPatient)+len(healthyCVPatient),len(idxs)))
    i=0
    for patient in (healthyCVPatient):
        imageList = listdir('trainROI/healthy/'+patient)
        HOGvec=[]
        count=0
        j=0
        for image in imageList:
            if "._" not in image:
#             read image from Healthy class
                img = imread('trainROI/healthy/'+patient+'/'+image)
                [h,w,d] = img.shape
                img = rescale(img, resize, anti_aliasing=True)
                [h,w,d] = img.shape
                
#                 Trim the image to a bucketable size
                F    = img[0:int(p*math.floor(h/p)), 0:int(p*math.floor(w/p)),:]
                
#               Extract HOG Features
                fd = hog(F[:,:,1], orientations=8, pixels_per_cell=(p, p),cells_per_block=(1, 1))
                
#              Reorganize features so that each index matches a sample
                features=np.reshape(fd,(((F.shape[1]*F.shape[0])/(p*p)),8))
                if count==0:
                    HOGvec=features
                else:
                    HOGvec=np.vstack((HOGvec,features))
                count=count+1
#       For each of the words in the inital dictionary, calculate the 
#       L2-distance between the features of the current image. If the distance
#       is too small too many times, remove the word from the dictionary.
        
        initWordsIdx=0
        for num in range(len(idxs)): 
            num_of_matches=0
            iters=0
            for n in HOGvec: 
                iters=iters+1
                r=np.linalg.norm(init_words[num]-n)
                if r < min_dist:
                    num_of_matches=num_of_matches+1
                iters=iters+1
            validationMatrix[i,j]=num_of_matches
            j=j+1
        i=i+1
    for patient in (unhealthyCVPatient):
        imageList = listdir('trainROI/unlabel/'+patient)
        HOGvec=[]
        count=0
        j=0
        for image in imageList:
            if "._" not in image:
#             read image from Healthy class
                img = imread('trainROI/unlabel/'+patient+'/'+image)
                [h,w,d] = img.shape
                img = rescale(img, resize, anti_aliasing=True)
                [h,w,d] = img.shape
                
#                 Trim the image to a bucketable size
                F    = img[0:int(p*math.floor(h/p)), 0:int(p*math.floor(w/p)),:]
                
#               Extract HOG Features
                fd = hog(F[:,:,1], orientations=8, pixels_per_cell=(p, p),cells_per_block=(1, 1))
                
#              Reorganize features so that each index matches a sample
                features=np.reshape(fd,(((F.shape[1]*F.shape[0])/(p*p)),8))
                if count==0:
                    HOGvec=features
                else:
                    HOGvec=np.vstack((HOGvec,features))
                count=count+1
#       For each of the words in the inital dictionary, calculate the 
#       L2-distance between the features of the current image. If the distance
#       is too small too many times, remove the word from the dictionary.
        
        initWordsIdx=0
        for num in range(len(idxs)): 
            num_of_matches=0
            iters=0
            for n in HOGvec: 
                iters=iters+1
                r=np.linalg.norm(init_words[num]-n)
                if r < min_dist:
                    num_of_matches=num_of_matches+1
                iters=iters+1
            validationMatrix[i,j]=num_of_matches
            j=j+1
        i=i+1
    
    
    
   
    normalizedXtrain=normalize(featureMatrix)
    normalizedXCV=normalize(validationMatrix)

    clf = svm.OneClassSVM(nu=nue, kernel="rbf", gamma=gammae)
    clf.fit(normalizedXtrain)
    y_CV=np.ones(shape=(len(unhealthyCVPatient)+len(healthyCVPatient),1))
    y_CV[len(healthyCVPatient):len(unhealthyCVPatient)+len(healthyCVPatient)]=-1
    y_pred_train = clf.predict(normalizedXtrain)
    y_pred_CV = clf.predict(normalizedXCV)

    return [y_pred_train, y_pred_CV, y_CV]
#        
#    for patients in range(len(unhealthyTestPatient)):
#        patientFeatures=[]
#        for images in listdir('testROI/healthy/'+healthyTestPatient[patients]):
#                    img = cv.imread(files)
#                    [h,w,d] = img.shape
#                    img = cv.resize(img, dsize=(int(math.floor(resize*h)), int(math.floor(resize*w))), interpolation=cv.INTER_CUBIC)
#                    [h,w,d] = img.shape
#
#                    F    = img[0:int(p*math.floor(h/p)), 0:int(p*math.floor(w/p)),:]
#                    fd = hog(F[:,:,1], orientations=8, pixels_per_cell=(p, p),cells_per_block=(1, 1))
#                    features=np.reshape(fd,(((F.shape[1]*F.shape[0])/(p*p)),8))
#                    patientFeatures.extend(features)
#        for n in features: 
#            
#            r=np.linalg.norm(rows-n)
#
#            if len(passed_thresh[0])<=thresh:
#                final_words.append(rows)
#                featureVector.append(len(passed_thresh[0]))
#                    
    #        [featureVector,hogVisualization] = extractHOGFeatures(im)  
    #      features=[features featureVector] 
    #      end
    #  [idx,C]  = kmeans(features,k) 
    #  
     
    #   ds=imageDatastore('trainROI/unhealthy', 'IncludeSubfolders',true) 
    #  bag = bagOfFeatures(ds,'VocabularySize',k,'GridStep',100) 
    #
    #   bagOfFeatures(ds)
    #for i in range(len(C))
    #    
    #end
    #         
if __name__ == '__main__': 


    params = pickle.load( open( 'h1.pkl', 'rb' ) )
    

    pars=[]

    output = [] 

    for row in params:
        pars.append(row)
        p=row[0]
        k=row[1]
        min_dist=row[2]
        threshold=row[3]
        gammae=row[4]
        nue=row[5]
        k=k.astype(np.int64)
        p=p.astype(np.int64)
        threshold=threshold.astype(np.int64)
        output.append(runROIDetector(p,k,min_dist,threshold, gammae, nue))
    pickle.dump( output, open( 'results1.pkl', 'wb' ) )