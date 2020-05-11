/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

// #define MATSIZE 50
// #define NRA MATSIZE            /* number of rows in matrix A */
// #define NCA MATSIZE            /* number of columns in matrix A */
// #define NCB MATSIZE            /* number of columns in matrix B */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


class MatrixFactory
{
public:
    MatrixFactory() {}
    ~MatrixFactory() {}

    void GetSquareRandomMatrix(float* arr, int m)
    {
        int count = 0;
        for (int i = 0; i < m * m; i++) {
            arr[i] = rand() % m;
        }
    }

    void GetC2Matrix(float* arr, int m)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i * (m - 1) + j] = 17.0 / (2.0 * (i + 1.0) + j + 1.0);
            }
        }
    }

    void fill_array(float* arr, const int n, const int m) {
        int count = 0;
        for (int i = 0; i < n * m; i++) {
            //for (int j = 0; j < m; j++) {
            arr[i] = count++;
            //}
        }
    }
};

class VectorFactory
{
public:
    VectorFactory() {}
    ~VectorFactory() {}

    void GetBVector(float* arr, int m)
    {
        for (int i = 0; i < m; i++)
        {
            arr[i] = 17.0 / (i * i + 1.0);
        }

        for (int i = m; i < m * m; i++)
        {
            arr[i] = 0.0;
        }
    }

    void GetRandomVector(float* arr, int m)
    {

        for (int i = 0; i < m; i++)
        {
            arr[i] = rand() % m;
        }

        for (int i = m; i < m * m; i++)
        {
            arr[i] = 0.0;
        }
    }
};

int matrix_Mult(float* aa, float* bb, float* cc, const int m)
{
    cout << "Mult.." << endl;
    //cout<< "A[0][0] = " << aa[0] << endl;
    int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */
double	a[m][m],           /* matrix A to be multiplied */
	b[m][m],           /* matrix B to be multiplied */
	c[m][m];           /* result matrix C */
MPI_Status status;

//MPI_Init(&argc,&argv);

MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;


/**************************** master task ************************************/
   if (taskid == MASTER)
   {
      printf("mpi_mm has started with %d tasks.\n",numtasks);
      // printf("Initializing arrays...\n");
      for (i=0; i<m; i++)
         for (j=0; j<m; j++)
            a[i][j]= aa[i*(m-1) + j];
      for (i=0; i<m; i++)
         for (j=0; j<m; j++)
            b[i][j]= bb[i*(m-1) + j];

      /* Measure start time */
      double start = MPI_Wtime();

      /* Send matrix data to the worker tasks */
      averow = m/numworkers;
      extra = m%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= extra) ? averow+1 : averow;   	
         // printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset][0], rows*m, MPI_DOUBLE, dest, mtype,
                   MPI_COMM_WORLD);
         MPI_Send(&b, m*m, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset][0], rows*m, MPI_DOUBLE, source, mtype, 
                  MPI_COMM_WORLD, &status);
         // printf("Received results from task %d\n",source);
      }

      /* Print results */
      /*
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      for (i=0; i<NRA; i++)
      {
         printf("\n"); 
         for (j=0; j<NCB; j++) 
            printf("%6.2f   ", c[i][j]);
      }
      printf("\n******************************************************\n");
      */

      /* Measure finish time */
      double finish = MPI_Wtime();
      printf("Done in %f seconds.\n", finish - start);
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, rows*m, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b, m*m, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

      for (k=0; k<m; k++)
         for (i=0; i<rows; i++)
         {
            c[i][k] = 0.0;
            for (j=0; j<m; j++)
               c[i][k] = c[i][k] + a[i][j] * b[j][k];
         }
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c, rows*m, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }
   for (int i = 0; i < m; i++)
   {
       for (int j = 0; j < m; j++)
       {
           cc[i*(m-1) + j] = c[i][j];
       }
       
   }
   
   //MPI_Finalize();
}

void NumberMatrixMult(float number, float* a, float* c, const int m)
{
    for (int i = 0; i < m * m; i++)
    {
        c[i] = a[i] * number;
    }
}

void MatrixSum(float* a, float* b, float* c, const int m)
{
    for (int i = 0; i < m * m; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv)
{   
    int row = 50;
    // cout << "N: ";
    // cin >> row;
    int col = row;

    float* matrixA = new float[row * col];
    float* vectorB = new float[row * col];
    float* matrixA1 = new float[row * col];
    float* vectorB1 = new float[row * col];
    float* vectorC1 = new float[row * col];
    float* matrixA2 = new float[row * col];
    float* matrixB2 = new float[row * col];
    float* matrixC2 = new float[row * col];

    MatrixFactory matrixFactory;
    matrixFactory.GetSquareRandomMatrix(matrixA, row);
    matrixFactory.GetSquareRandomMatrix(matrixA1, row);
    matrixFactory.GetSquareRandomMatrix(matrixB2, row);
    matrixFactory.GetSquareRandomMatrix(matrixA2, row);
    matrixFactory.GetC2Matrix(matrixC2, row);

    VectorFactory vectorFactory;
    vectorFactory.GetBVector(vectorB, row);
    vectorFactory.GetRandomVector(vectorB1, row);
    vectorFactory.GetRandomVector(vectorC1, row);

    MPI_Init(&argc, &argv);
    
    float* vectorY1 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(matrixA, vectorB, vectorY1, row);
    //MPI_Finalize();

    float* multForY2 = new float[row * col];
    NumberMatrixMult(17.0, vectorB, multForY2, row);

    float* sumForY2 = new float[row * col];
    MatrixSum(multForY2, vectorC1, sumForY2, row);

    float* vectorY2 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(matrixA1, sumForY2, vectorY2, row);
    //MPI_Finalize();

    float* sumForM3 = new float[row * col];
    MatrixSum(matrixB2, matrixC2, sumForM3, row);

    float* matrixY3 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(matrixA2, sumForM3, matrixY3, row);
    //MPI_Finalize();

    float* matrixResult = new float[row * col];

    float* L11 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(matrixY3, matrixY3, L11, row);
    //MPI_Finalize();

    float* L21 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(L11, matrixY3, L21, row);
    //MPI_Finalize();

    float* L12 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(vectorY2, vectorY1, L12, row);
    //MPI_Finalize();

    float* L22 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(L12, matrixY3, L22, row);
    //MPI_Finalize();

    float* L31 = new float[row * col];
    MatrixSum(L21, L22, L31, row);

    float* L13 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(matrixY3, vectorY2, L13, row);
    //MPI_Finalize();

    float* L23 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(vectorY1, vectorY2, L23, row);
    //MPI_Finalize();

    float* L24 = new float[row * col];
    //MPI_Init(&argc, &argv);
    matrix_Mult(L13, vectorY1, L24, row);
    //MPI_Finalize();

    float* L32 = new float[row * col];
    MatrixSum(L23, L24, L32, row);

    MatrixSum(L31, L32, matrixResult, row);
    MPI_Finalize();

    cout << "******************************" << endl;
    cout << "***** Results: *****" << endl;
    cout << "******************************" << endl;

    for (int i = 0; i < row; i++)
    {
        cout << L32[i] << endl;
    }

}
