
import os
import sys
import numpy as np
from PIL import Image

# function to read images from a folder.


def read_images (path , sz= None ):
    c = 1
    X,y = [], []
    for dirname , dirnames , filenames in os. walk ( path ):
        for subdirname in dirnames :
            subject_path = os. path . join ( dirname , subdirname )
            #print(subject_path)
            fileNames = os. listdir ( subject_path )
            
            for i in range(1):
                try :
                    im = Image . open (os. path . join ( subject_path , fileNames[i] ))
                    im = im. convert ("L")
                    # resize to given size (if given )
                    if (sz is not None ):
                        im = im. resize (sz , Image . ANTIALIAS )
                    X. append (np. asarray (im , dtype =np. uint8 ))
                    y. append (c)
                except IOError :
                    print ("I/O error")
                except :
                    print("Unexpected error :", sys . exc_info () [0])
                    raise
                c = c+1
        if c > 30 :
            break
    return [X,y]

def read_image(filePath, sz=None):
    imageAsArray = None
    try :
        im = Image.open(filePath)
        im = im. convert ("L")
        # resize to given size (if given )
        if (sz is not None ):
            im = im. resize (sz , Image . ANTIALIAS )
        imageAsArray = np. asarray (im , dtype =np. uint8 )
    except IOError :
        print ("I/O error")
    except :
        print("Unexpected error :", sys . exc_info () [0])
        raise
    return imageAsArray

# A function to return row matrix
def asRowMatrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0]. size ), dtype =X [0]. dtype )
    for row in X:
        mat = np. vstack (( mat , np. asarray ( row ). reshape (1 , -1)))
        return mat

# A function to return column matrix
def asColumnMatrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((X [0]. size , 0) , dtype =X [0]. dtype )
    for col in X:
        mat = np. hstack (( mat , np. asarray ( col ). reshape ( -1 ,1)))
        return mat

# A function to determine eigen faces using Principal component
# analysis. Eigen faces are the features used to represent a face.
def pca (X, y, num_components =0) :
    [n,d] = X. shape
    if ( num_components <= 0) or ( num_components >n):
        num_components = n
    mu = X. mean ( axis =0)
    X = X - mu
    if n>d:
        C = np. dot (X.T,X)
        [ eigenvalues , eigenvectors ] = np. linalg . eigh (C)
    else :
        C = np. dot (X,X.T)
        [ eigenvalues , eigenvectors ] = np. linalg . eigh (C)
        eigenvectors = np. dot (X.T, eigenvectors )
        #for i in range(n):
         #   eigenvectors [:,i] = eigenvectors [:,i]/ np. linalg . norm ( eigenvectors [:,i])
    # or simply perform an economy size decomposition
    eigenvectors , eigenvalues , variance = np. linalg . svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np. argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ]. copy ()
    eigenvectors = eigenvectors [: ,0: num_components ]. copy ()
    return [ eigenvalues , eigenvectors , mu]

# A function to project a ve
def project (W, X, mu= None ):
    if mu is None :
        return np. dot (X,W)
    return np. dot (X - mu , W)


# An abstract distance class

class AbstractDistance ( object ):
    def __init__ (self , name ):
        self . _name = name
    def __call__ (self ,p,q):
        raise NotImplementedError (" Every AbstractDistance must implement the __call__method .")
    
    @property
    def name ( self ):
        return self . _name
    
    def __repr__ ( self ):
        return self . _name
    
# An euclidean distance class
class EuclideanDistance ( AbstractDistance ):
    def __init__ ( self ):
        AbstractDistance . __init__ (self ," EuclideanDistance ")
    def __call__ (self , p, q):
        p = np. asarray (p). flatten ()
        q = np. asarray (q). flatten ()
        return np. sqrt (np. sum (np. power ((p-q) ,2)))

#A Cosine distance class
class CosineDistance ( AbstractDistance ):
    def __init__ ( self ):
        AbstractDistance . __init__ (self ," CosineDistance ")
    def __call__ (self , p, q):
        p = np. asarray (p). flatten ()
        q = np. asarray (q). flatten ()
        return -np.dot(p.T,q) / (np. sqrt (np.dot(p,p.T)*np.dot(q,q.T)))

# A base model for face classifier
class BaseModel ( object ):
    def __init__ (self , X=None , y=None, dist_metric = EuclideanDistance () , num_components=0) :
        self . dist_metric = dist_metric
        self . num_components = 0
        self . projections = []
        self .W = []
        self .mu = []
        if (X is not None ) and (y is not None ):
            self . compute (X,y)
    
    def compute (self , X, y):
        raise NotImplementedError (" Every BaseModel must implement the compute method .")
    
    def predict (self , X):
        minDist = 10000000000
        minClass = -1
        Q = project ( self .W, X. reshape (1 , -1) , self .mu)
        for i in range (len( self . projections )):
            dist = self . dist_metric ( self . projections [i], Q)
            if dist < minDist :
                minDist = dist
                minClass = self .y[i]
        return minClass
 
    def euc_distance(self, X, Y):
        Q = project ( self .W, X. reshape (1 , -1) , self .mu)
        W = project ( self .W, Y. reshape (1 , -1) , self .mu)
        dist = self . dist_metric (Q,W)
        return dist
    
# An eigenfaces model
class EigenfacesModel ( BaseModel ):
    def __init__ (self , X=None , y=None, dist_metric = EuclideanDistance () , num_components=0) :
        super ( EigenfacesModel , self ). __init__ (X=X,y=y, dist_metric = dist_metric , num_components = num_components )

    def compute (self,X, y):
        [D, self .W, self .mu] = pca ( asRowMatrix (X),y, self . num_components )
        # store labels
        self .y = y
        # store projections
        for xi in X:
            self . projections . append ( project ( self .W, xi. reshape (1 , -1) , self .mu))



'''
#  Testing the face recogniser 

# read images 
[X, y]  = read_images("C:\\Arka")

# compute the eigenfaces model
model = EigenfacesModel (X[0:] , y [0:],num_components = 30)
print(model.W.shape)

newImage = read_image("C:/Arka/s8/10.pgm")

newImage2 = read_image("C:/Test/Test/s39/1.pgm")

print("eulidean distance " , model.euc_distance(newImage, newImage2))

# get a prediction for the first observation
print(" expected =", y[0] , "/", " predicted =", model . predict (newImage2))
print(" expected =", y[29] , "/", " predicted =", model . predict (X [0]))

'''   

