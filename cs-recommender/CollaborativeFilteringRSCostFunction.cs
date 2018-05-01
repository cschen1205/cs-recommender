using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.ProblemModels;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Recommender
{
    public class CollaborativeFilteringRSCostFunction : CostFunction
    {
        public Matrix<double> mY;
        protected int[,] mR;
        protected Matrix<double> mR_matrix;
        protected int mSampleCount = 0;
        protected int mFeatureNum = 0;

        protected double mRegularizationLambda = 0;

        public double RegularizationLambda
        {
            get { return mRegularizationLambda; }
            set { mRegularizationLambda = value; }
        }

        public CollaborativeFilteringRSCostFunction(Matrix<double> Y, int[,] r, int sample_count, int X_dimension_count, int dimension_count)
            : base(dimension_count, -1000000, 100000)
        {
            mY = Y;
            mR = r;

            mFeatureNum = X_dimension_count;

            mSampleCount = sample_count;

            int n_m=mSampleCount;
            int n_u=Y.ColumnCount;
            mR_matrix = new SparseMatrix(n_m, n_u);
            for (int i = 0; i < n_m; ++i)
            {
                for (int j = 0; j < n_u; ++j)
                {
                    mR_matrix[i, j] = mR[i, j];
                }
            }
        }

        public static void RollVectorIntoMatrix(double[] theta_x, int theta_row_count, int theta_col_count, int X_row_count, int X_col_count, Matrix<double> Theta, Matrix<double> X)
        {
           

            for (int j = 0; j < theta_row_count; ++j)
            {
                for (int k = 0; k < theta_col_count; ++k)
                {
                    int index = j * theta_col_count + k;
                    Theta[j, k] = theta_x[index];
                }
            }

            int theta_dimension = theta_row_count * theta_col_count;

            for (int i = 0; i < X_row_count; ++i)
            {
                for (int k = 0; k < X_col_count; ++k)
                {
                    int index = i * X_col_count + k + theta_dimension;
                    X[i, k] = theta_x[index];
                }
            }
        }

        public static void UnrollMatrixIntoVector(Matrix<double> Theta, Matrix<double> X, double[] theta_x)
        {
            int theta_row_count = Theta.RowCount;
            int theta_col_count = Theta.ColumnCount;

            int X_row_count = X.RowCount;
            int X_col_count = X.ColumnCount;

            for (int j = 0; j < theta_row_count; ++j)
            {
                for (int k = 0; k < theta_col_count; ++k)
                {
                    int index = j * theta_col_count + k;
                    theta_x[index] = Theta[j, k];
                }
            }

            int theta_dimension = theta_row_count * theta_col_count;

            for (int i = 0; i < X_row_count; ++i)
            {
                for (int k = 0; k < X_col_count; ++k)
                {
                    int index = i * X_col_count + k + theta_dimension;
                    theta_x[index] = X[i, k];
                }
            }
        }

        public override double Evaluate(double[] theta_x)
        {
            int n_u = mY.ColumnCount;
            int n_m = mSampleCount;

            Matrix<double> Theta = new DenseMatrix(n_u, mFeatureNum);
            Matrix<double> X = new DenseMatrix(n_m, mFeatureNum);
            RollVectorIntoMatrix(theta_x, n_u, mFeatureNum, n_m, mFeatureNum, Theta, X);

            Matrix<double> XThetaPrime = X.Multiply(Theta.Transpose());

            Matrix<double> XThetaPrimeMinusY = XThetaPrime - mY;

            double J = 0;
            for (int i = 0; i < n_m; ++i)
            {
                for (int j = 0; j < n_u; ++j)
                {
                    if (mR[i, j] == 1)
                    {
                        J += System.Math.Pow(XThetaPrimeMinusY[i, j], 2);
                    }
                }
            }
            J /= 2;

            if (mRegularizationLambda != 0)
            {
                for (int j = 0; j < n_u; ++j)
                {
                    for (int k = 0; k < mFeatureNum; ++k)
                    {
                        J += (mRegularizationLambda * System.Math.Pow(Theta[j, k], 2)) / 2;
                    }
                }

                for (int i = 0; i < n_m; ++i)
                {
                    for (int k = 0; k < mFeatureNum; ++k)
                    {
                        J += (mRegularizationLambda * System.Math.Pow(X[i, k], 2)) / 2;
                    }
                }
            }
            return J;
        }



        protected override void _CalcGradient(double[] theta_x, double[] grad)
        {
            int n_u = mY.ColumnCount;
            int n_m = mSampleCount;

            Matrix<double> Theta = new DenseMatrix(n_u, mFeatureNum);
            Matrix<double> X = new DenseMatrix(n_m, mFeatureNum);

            RollVectorIntoMatrix(theta_x, n_u, mFeatureNum, n_m, mFeatureNum, Theta, X);

            
            Matrix<double> XThetaPrime = X.Multiply(Theta.Transpose());

            
            Matrix<double> XThetaPrimeMinusY = (XThetaPrime - mY).PointwiseMultiply(mR_matrix);

            
            Matrix<double> Grad_Theta = XThetaPrimeMinusY.Transpose().Multiply(X) +Theta.Multiply(mRegularizationLambda);

            
            Matrix<double> Grad_X = XThetaPrimeMinusY.Multiply(Theta) +X.Multiply(mRegularizationLambda);

            UnrollMatrixIntoVector(Grad_Theta, Grad_X, grad);
        }
    }
}
