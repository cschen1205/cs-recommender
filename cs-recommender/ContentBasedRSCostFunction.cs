using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.ProblemModels;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Recommender
{
    public class ContentBasedRSCostFunction : CostFunction
    {
        public Matrix<double> mX;
        public Matrix<double> mY;
        public int[,] mR;
        protected int mSampleCount = 0;

        protected double mRegularizationLambda = 0;

        public double RegularizationLambda
        {
            get { return mRegularizationLambda; }
            set { mRegularizationLambda = value; }
        }

        public ContentBasedRSCostFunction(Matrix<double> X, Matrix<double> Y, int[,] r, int sample_count, int dimension_count)
            : base(dimension_count, -1000000, 100000)
        {
            mX = X;
            mY = Y;
            mR = r;

            mSampleCount = sample_count;
        }

        public override double Evaluate(double[] theta)
        {
            double J = 0;

            int n_u = mY.ColumnCount;
            int n_m = mSampleCount;

            for (int j = 0; j < n_u; ++j)
            {
                for (int i = 0; i < n_m; ++i)
                {
                    if (mR[i, j] == 1)
                    {
                        double sum = 0;
                        for (int d2 = 0; d2 < mX.ColumnCount; ++d2)
                        {
                            int index = j * mX.ColumnCount + d2;
                            sum += mX[i, d2] * theta[index];
                        }
                        J += System.Math.Pow(sum - mY[i, j], 2);
                    }
                }
            }

            J /= 2;

            for (int j = 0; j < mY.ColumnCount; ++j)
            {
                for (int d2 = 1; d2 < mX.ColumnCount; ++d2)
                {
                    int d = j * mX.ColumnCount + d2;
                    J += (mRegularizationLambda * theta[d] * theta[d]) / 2;
                }
            }

            return J;
        }



        protected override void _CalcGradient(double[] theta, double[] grad)
        {
            for (int d1 = 0; d1 < mY.ColumnCount; ++d1)
            {
                for (int d2 = 0; d2 < mX.ColumnCount; ++d2)
                {
                    int d = d1 * mX.ColumnCount + d2;
                    grad[d] = 0;

                    for (int i = 0; i < mSampleCount; ++i)
                    {
                        double sum = 0;
                        for (int d3 = 0; d3 < mX.ColumnCount; ++d3)
                        {
                            sum += mX[i, d3] * theta[d1 * mX.ColumnCount + d3];
                        }
                        grad[d] += (sum - mY[i, d1]) * mX[i, d2];
                    }

                    if (d2 != 0)
                    {
                        grad[d] += (mRegularizationLambda * theta[d]);
                    }
                }
            }
        }
    }
}
