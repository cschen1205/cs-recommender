using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.LocalSearch;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using ContinuousOptimization;

namespace Recommender
{
    public class ContentBasedRS<T> 
        where T: RatedItem
    {
        protected SingleTrajectoryContinuousSolver mLocalSearcher = new GradientDescent();
        protected int mMaxSolverIteration = 2000;
        protected double[] mTheta = null;

        protected double mRegularizationLambda = 0;

        protected bool mEnsureNormalEquationInvertibility = false;
        protected double mNormalEquationInventibilityLambda = 0.0001;

        public SingleTrajectoryContinuousSolver LocalSearch
        {
            get { return mLocalSearcher; }
            set { mLocalSearcher = value; }
        }

        public bool EnsureNormalEquationInvertibility
        {
            get { return mEnsureNormalEquationInvertibility; }
            set { mEnsureNormalEquationInvertibility = value; }
        }

        public double NormalEquationInventibilityLambda
        {
            get { return mNormalEquationInventibilityLambda; }
            set { mNormalEquationInventibilityLambda = value; }
        }

        public double RegularizationLambda
        {
            get { return mRegularizationLambda; }
            set { mRegularizationLambda = value; }
        }

        public enum CalcMode
        {
            GradientDescent,
            NormalEquation
        }

        protected CalcMode mCalcMode = CalcMode.GradientDescent;

        public CalcMode Mode
        {
            get { return mCalcMode; }
            set { mCalcMode = value; }
        }

        public int MaxLocalSearchIteration
        {
            get { return mMaxSolverIteration; }
            set { mMaxSolverIteration = value; }
        }

        public void PredictRank(T rec)
        {
            bool[] r=rec.IsRated;

            for (int d1 = 0; d1 < rec.UserRanks.Length; ++d1)
            {
                double predicted_rank = 0;
                if (!r[d1])
                {
                    for (int d2 = 0; d2 < rec.X.Length + 1; ++d2)
                    {
                        int d = d1 * (rec.X.Length + 1) + d2;
                        if (d2 == 0)
                        {
                            predicted_rank += mTheta[d];
                        }
                        else
                        {
                            predicted_rank += mTheta[d] * rec.X[d2 - 1];
                        }
                    }
                    r[d1] = true;
                    rec.UserRanks[d1] = predicted_rank;
                }
            }
        }

        

        public void Compute(List<T> data_set)
        {
            int n_m = data_set.Count;
            if (n_m == 0)
            {
                throw new ArgumentException("Data set is empty!");
            }
            int x_dimension = data_set[0].X.Length;
            int n_u = data_set[0].UserRanks.Length;

            Matrix<double> X_matrix = new SparseMatrix(n_m, x_dimension+1, 0);
            Matrix<double> Y_matrix = new SparseMatrix(n_m, n_u, 0);
            int[,] r_matrix = new int[n_m, n_u];

            int dimension=n_u * (x_dimension + 1);

            for (int i = 0; i < n_m; ++i)
            {
                X_matrix[i, 0] = 1;
                T rating = data_set[i];
                double[] X = rating.X;
                double[] Y = rating.UserRanks;
                bool[] r = rating.IsRated;
                
                for (int d = 0; d < x_dimension; ++d)
                {
                    X_matrix[i, d+1] = X[d];
                }

                for (int d = 0; d < n_u; ++d)
                {
                    Y_matrix[i, d] = Y[d];
                    r_matrix[i, d] = r[d] ? 1 : 0;
                }
            }

            if (mCalcMode == CalcMode.GradientDescent)
            {
                ContentBasedRSCostFunction f = new ContentBasedRSCostFunction(X_matrix, Y_matrix, r_matrix, n_m, dimension);
                f.RegularizationLambda = mRegularizationLambda;
                double[] theta_0 = new double[dimension];
                for (int d = 0; d < dimension; ++d)
                {
                    theta_0[d] = 0;
                }
                ContinuousSolution solution = mLocalSearcher.Minimize(theta_0, f, mMaxSolverIteration);

                mTheta = solution.Values;
            }
            else if (mCalcMode == CalcMode.NormalEquation)
            {
                Matrix<double> X_transpose_matrix = X_matrix.Transpose();
                Matrix<double> X_transpose_X_matrix = X_transpose_matrix.Multiply(X_matrix);
                Matrix<double> X_transpose_Y_matrix = X_transpose_matrix.Multiply(Y_matrix);

                Matrix<double> X_transpose_X_inverse_matrix = null;
                if (mEnsureNormalEquationInvertibility)
                {
                    Matrix<double> lambda_matrix = SparseMatrix.Identity(dimension);
                    for (int d = 0; d < x_dimension+1; ++d)
                    {
                        lambda_matrix[d, d] = (d == 0) ? 0 : mNormalEquationInventibilityLambda;
                    }
                    X_transpose_X_inverse_matrix = (X_transpose_X_matrix + lambda_matrix).Inverse();
                }
                else
                {
                    X_transpose_X_inverse_matrix = X_transpose_X_matrix.Inverse();
                }
                Matrix<double> theta_matrix = X_transpose_X_inverse_matrix.Multiply(X_transpose_Y_matrix);
                mTheta = new double[dimension];
                for (int d1 = 0; d1 < n_u; ++d1)
                {
                    for (int d2 = 0; d2 < x_dimension + 1; ++d2)
                    {
                        int index = d1 * (x_dimension + 1) + d2;
                        mTheta[index] = theta_matrix[d2, d1];
                    }
                }
            }

            for (int i = 0; i < n_m; ++i)
            {
                PredictRank(data_set[i]);
            }
        }

        public double ComputeCost(List<T> data_set)
        {
            int sample_count = data_set.Count;
            if (sample_count == 0)
            {
                throw new ArgumentException("data set is empty!");
            }
            int x_dimension = data_set[0].X.Length;
            int y_dimension = data_set[0].UserRanks.Length;

            Matrix<double> X_matrix = new SparseMatrix(sample_count, x_dimension + 1, 0);
            Matrix<double> Y_matrix = new SparseMatrix(sample_count, y_dimension, 0);
            int[,] r_matrix = new int[sample_count, y_dimension];

            int dimension = y_dimension * (x_dimension + 1);

            for (int i = 0; i < sample_count; ++i)
            {
                X_matrix[i, 0] = 1;
                T rating = data_set[i];
                double[] X = rating.X;
                double[] Y = rating.UserRanks;
                bool[] r = rating.IsRated;

                for (int d = 0; d < x_dimension; ++d)
                {
                    X_matrix[i, d + 1] = X[d];
                }

                for (int d = 0; d < y_dimension; ++d)
                {
                    Y_matrix[i, d] = Y[d];
                    r_matrix[i, d] = r[d] ? 1 : 0;
                }
            }

            ContentBasedRSCostFunction f = new ContentBasedRSCostFunction(X_matrix, Y_matrix, r_matrix, sample_count, dimension);
            f.RegularizationLambda = 0;
            return f.Evaluate(mTheta);
        }
    }
}
