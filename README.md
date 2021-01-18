# cuSparse

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software. Also included are test data and scripts to verify the installation was successful.

Environment Requirements
------------------------

**Programming Language:** CUDA C/C++ (tested using cuda/10.2.89)

Installation Instructions
-------------------------
For unified memory implementation 

(1) In the ```Makefile```, edit the variable ```CUDA_INSTALL_PATH``` to match the CUDA installation directory and ```CUDA_SAMPLES_PATH``` to match the directory of CUDA samples that were installed with CUDA.

(2) Add ```<CUDA installation directory>/bin``` to the system's $PATH environment variable.

(3) Type ```make``` to compile the ./sptrsv

Test Cases
----------

* To verify the build was successful, a small test can be done: 
	* ```cd``` into where ```./sptrsv``` is. 
	* run ```run.sh ``` to test SpTRSV on a input matrix ash85.mtx. If the test was successful, it should output the test pass and run time without any error message.
	* edit the task number and gpu number in run.sh to test on multiple GPU and define the test number.

User Guide
==========

#### Using the ./sptrsv 
	
 	```./sptrsv -mtx [A.mtx]```
    * ```[A.mtx]```: It will load input matrix from the given path
    
#### Output

The ./sptrsv will run tests based on imput matrix with options specified by users (in the common file). The exection time will be reported. The correctness of the output of SpTRSV are verified by comparing their results with the x_ref.
    
