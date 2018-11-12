# ssvm_python
SSVM is a reformulation of conventional SVM and can be solved by a fast Newton-Armijo algorithm

This project use **matlab engine for python** to call matlab function, and create a sklearn-like way to use those functions

<p align="center"><img src="/image/ssvm_architecture.png" height="200"></p>

## Functions
After import ```ssvm.py``` package, you can use ```Smooth Support Vector Classification(SSVC)``` & ```Smooth Support Vector Regression(SSVR)``` with function below:
- ```.fit(data, label)```
- ```.predict(data)```
- ```.score(data, label)```
- ```.print_params()```
- ```.get_params()```

just like sklearn, very easy, right:-)?

## Usage
You can use SSVC to classification or SSVR to do regression(**notice: both linear-only**).
#### 1. First, download these files and move to your workpath
`ssvc.m`、`ssvr.m`、`ssvm.py`
#### 2. In your python code, import package
- if you want to use SSVC, use ```from ssvm import SSVC```
- if you want to use SSVR, use  ```from ssvm import SSVR```
#### 3. Create ssvc/ssvr instance
- ```ssvc = SSVC()``` or ```ssvr = SSVR()```
#### 4. Train data
- ```ssvc.fit(data, label)``` or ```ssvr.fit(data, value)```

notice that about input format:
- data: shape must be (m, n) array-like type, which m is data size and n is feature number
- label/value: shape must be (1, m) or (m, 1) array-like type, which m is data size
#### 5.predict
use ```.predict(data)``` to predict

## Demo
you can see demo code in `SSVC_Demo.ipynb` & `SSVR_Demo.ipynb`

## References
### paper 
- [SSVM: A Smooth Support Vector Machine for
Classification
](http://jupiter.math.nctu.edu.tw/~yuhjye/assets/file/publications/journal_papers/J18_SSVM%20A%20Smooth%20Support%20Vector%20Machine%20for%20Classification.pdf)
- [epsilon-SSVR: A Smooth Support Vector Machine for epsilon-insensitive Regression](https://pdfs.semanticscholar.org/6a1d/237dcbfbbeb66e0ce900f45119503f8ce8bb.pdf)

### matlab code
- [dsmilab/ssvm](https://github.com/dsmilab/ssvm/tree/master/src)
