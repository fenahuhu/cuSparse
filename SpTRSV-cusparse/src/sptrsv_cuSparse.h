#ifndef _SPTRSV_SYNCFREE_SERIALREF_
#define _SPTRSV_SYNCFREE_SERIALREF_

#include "common.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <assert.h>



int sptrsv_cuSparse(const int           *csrRowPtr,
                              const int           *csrColIdx,
                              const VALUE_TYPE    *csrVal,
                              const int            m,
                              const int            n,
                              const int            nnz,
                              const int            substitution,
                              const int            rhs,
                                    VALUE_TYPE    *x,
                              const VALUE_TYPE    *b,
                              const VALUE_TYPE    *x_ref)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }


    cudaSetDevice(0);
    
    
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;


    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
   

    status = cusparseCreate(&handle);
   

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    

    status = cusparseSetStream(handle, stream);
    



    //transfer csc to csr 
    
    
    // transfer host mem to device mem
    int *d_csrRowPtr;
    int *d_csrColIdx;
    double *d_csrVal;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;
    
    // Matrix csr
    cudaMallocManaged((void **)&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMallocManaged((void **)&d_csrColIdx, nnz  * sizeof(int));
    cudaMallocManaged((void **)&d_csrVal,    nnz  * sizeof(double));
    cudaMallocManaged((void **)&d_b, n * rhs * sizeof(VALUE_TYPE));
    cudaMallocManaged((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz  * sizeof(double),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, (size_t)(n * sizeof(VALUE_TYPE)), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, (size_t)(n * sizeof(VALUE_TYPE)));
    
    
    //  - SpTRSV Serial analyser start!
    printf(" - SpTRSV analyser start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    
cusparseMatDescr_t descr = 0;
csrsv2Info_t info = 0;
int pBufferSize;
void *pBuffer = 0;
int structural_zero;
int numerical_zero;
const double alpha = 1.;
const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans = CUSPARSE_OPERATION_TRANSPOSE;

// step 1: create a descriptor which contains
// - matrix L is base-1
// - matrix L is lower triangular
// - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
//   (L may not have all diagonal elements.)
cusparseCreateMatDescr(&descr);
cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

// step 2: create a empty info structure
cusparseCreateCsrsv2Info(&info);    
// step 3: query how much memory used in csrsv2, and allocate the buffer
cusparseDcsrsv2_bufferSize(handle, trans, m, nnz, descr, d_csrVal, d_csrRowPtr, d_csrColIdx, info, &pBufferSize);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
cudaMalloc((void**)&pBuffer, pBufferSize);

    
// step 4: perform analysis
cusparseDcsrsv2_analysis(handle, trans, m, nnz, descr,
    d_csrVal, d_csrRowPtr, d_csrColIdx,
    info, policy, pBuffer);
// L has unit diagonal, so no structural zero is reported.
status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
}

gettimeofday(&t2, NULL);
    double time_sptrsv_analyser = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("SpTRSV anlysis on L used %4.2f ms\n", time_sptrsv_analyser);
    
    struct timeval t3, t4;
    gettimeofday(&t3, NULL);
    
// step 5: solve L*y = x
cusparseDcsrsv2_solve(handle, trans, m, nnz, &alpha, descr,
   d_csrVal, d_csrRowPtr, d_csrColIdx, info,
   d_b, d_x, policy, pBuffer);
// L has unit diagonal, so no numerical zero is reported.
status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
}

// step 6: free resources
cudaFree(pBuffer);
cusparseDestroyCsrsv2Info(info);
cusparseDestroyMatDescr(descr);
cusparseDestroy(handle);

    gettimeofday(&t4, NULL);
    time_sptrsv_analyser = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    printf("SpTRSV solve on L used %4.2f ms\n", time_sptrsv_analyser);

    //  - SpTRSV Serial executor start!
   
     cudaMemcpy(x, d_x, (size_t)(n * sizeof(VALUE_TYPE)), cudaMemcpyDeviceToHost);
    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(d_x[i] - x_ref[i]);
        //  if (x_ref[i] != d_x[i]) 
	//printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], d_x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("SpTRSV Serial executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("SpTRSV Serial executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

      cudaFree(d_csrColIdx);
    	cudaFree(d_csrRowPtr);
    	cudaFree(d_csrVal);
    	cudaFree(d_b);
    	cudaFree(d_x);

    return 0;
}

#endif
