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
    public class CollaborativeFilteringRS<T>
        where T : RatedItem
    {
        protected SingleTrajectoryContinuousSolver mLocalSearcher = new GradientDescent();
        protected int mMaxSolverIteration = 2000;
        protected double[] mThetaX = null;
        protected Matrix<double> mTheta = null;

        protected double mRegularizationLambda = 0;

        public delegate void SteppedHandle(BaseSolution<double> solution, int step);
        public event SteppedHandle Stepped;
        public delegate void SolutionUpdatedHandle(BaseSolution<double> solution, int step);
        public event SolutionUpdatedHandle SolutionUpdated;

        public int MaxSolverIteration
        {
            get { return mMaxSolverIteration; }
            set { mMaxSolverIteration = value; }
        }

        protected void OnStepped(BaseSolution<double> solution, int step)
        {
            if (Stepped != null)
            {
                Stepped(solution, step);
            }
        }

        protected void OnSolutionUpdated(BaseSolution<double> best_solution, int step)
        {
            if (SolutionUpdated != null)
            {
                SolutionUpdated(best_solution, step);
            }
        }


        public CollaborativeFilteringRS()
        {
            mLocalSearcher.Stepped += (s, step) =>
                {
                    OnStepped(s, step);
                };
            mLocalSearcher.SolutionUpdated += (s, step) =>
                {
                    OnSolutionUpdated(s, step);
                };
        }


        public SingleTrajectoryContinuousSolver LocalSearch
        {
            get { return mLocalSearcher; }
            set
            {
                if (mLocalSearcher != value)
                {
                    mLocalSearcher = value;
                    value.Stepped += (s, k) =>
                        {
                            OnStepped(s, k);
                        };
                    value.SolutionUpdated += (s, k) =>
                        {
                            OnSolutionUpdated(s, k);
                        };
                }
            }
        }
        public void UndoMeanNormalization(List<T> data_set, double[] Ymean)
        {
            int n_m = data_set.Count;

            for (int i = 0; i < n_m; ++i)
            {
                T rec = data_set[i];
                double[] Y = rec.UserRanks;
                bool[] r = rec.IsRated;
                int n_u = Y.Length;

                for (int j = 0; j < n_u; ++j)
                {
                    Y[j] += Ymean[i];
                }
            }
        }

        public void DoMeanNormalization(List<T> data_set, out double[] YMean)
        {
            int n_m = data_set.Count;
            YMean = new double[n_m];

            for (int i = 0; i < n_m; ++i)
            {
                T rec = data_set[i];
                double[] Y = rec.UserRanks;
                bool[] r=rec.IsRated;
                double sum = 0;
                int count = 0;
                int n_u=Y.Length;
                
                for (int j = 0; j < n_u; ++j)
                {
                    if (r[j])
                    {
                        sum += Y[j];
                        count++;
                    }
                }
                double average = 0;

                if (count > 0)
                {
                    average = sum / count;
                }
                YMean[i] = average;

                for (int j = 0; j < n_u; ++j)
                {
                    if (r[j])
                    {
                        Y[j] -= average;
                    }
                }
            }
        }

        public double RegularizationLambda
        {
            get { return mRegularizationLambda; }
            set { mRegularizationLambda = value; }
        }

        public int MaxLocalSearchIteration
        {
            get { return mMaxSolverIteration; }
            set { mMaxSolverIteration = value; }
        }

        public List<T> SelectMostSimilar(T rec, List<T> data_set, int K)
        {
            List<T> similar_items = new List<T>();

            List<T> temp = OrderBySimilarity(rec, data_set);

            for (int i = 0; i < K; ++i)
            {
                similar_items.Add(temp[i]);
            }

            return similar_items;
        }

        public List<T> SelectHigestRanked(int user_index, List<T> data_set, int K)
        {
            List<T> highest_ranked_item = new List<T>();

            List<T> temp = OrderByRank(user_index, data_set);

            for (int i = 0; i < K; ++i)
            {
                highest_ranked_item.Add(temp[i]);
            }

            return highest_ranked_item;
        }

        public List<T> OrderByRank(int user_index, List<T> data_set)
        {
            int n_m = data_set.Count;
            List<T> temp = new List<T>();
            for (int i = 0; i < n_m; ++i)
            {
                temp.Add(data_set[i]);
            }

            temp.Sort((t1, t2) =>
            {
                return t2.UserRanks[user_index].CompareTo(t1.UserRanks[user_index]);
            });

            return temp;
        }

        public List<T> OrderBySimilarity(T rec, List<T> data_set)
        {
            int n_m = data_set.Count;
            List<T> temp = new List<T>();
            for (int i = 0; i < n_m; ++i)
            {
                temp.Add(data_set[i]);
            }

            temp.Sort((t1, t2) =>
            {
                return GetDistance(t1.X, rec.X).CompareTo(GetDistance(t2.X, rec.X));
            });

            return temp;
        }

        protected virtual double GetDistance(double[] x1, double[] x2)
        {
            int n = x1.Length;
            double sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum+=System.Math.Pow(x1[i]-x2[i], 2);
            }
            return System.Math.Sqrt(sum);
        }

        public delegate double NextRandomDoubleHandle();
        public event NextRandomDoubleHandle NextRandomDoubleRequested;

        protected Gaussian mRandom = new Gaussian(0, 1);
        protected virtual double NextRandomDouble()
        {
            if (NextRandomDoubleRequested != null)
            {
                return NextRandomDoubleRequested();
            }
            return mRandom.Next();
        }

        public Matrix<double> Compute(List<T> data_set, int num_features)
        {
            int n_m = data_set.Count;
            if (n_m == 0)
            {
                throw new ArgumentException("Data set is empty!");
            }
            int n_u = data_set[0].UserRanks.Length;

            Matrix<double> Y_matrix = new SparseMatrix(n_m, n_u, 0);
            int[,] r_matrix = new int[n_m, n_u];

            int theta_dimension = n_u * num_features;
            int content_dimension = n_m * num_features;

            int dimension = theta_dimension + content_dimension;

            for (int i = 0; i < n_m; ++i)
            {
                T rating = data_set[i];
                rating.ItemIndex = i;
                double[] Y = rating.UserRanks;
                bool[] r = rating.IsRated;

                for (int j = 0; j < n_u; ++j)
                {
                    Y_matrix[i, j] = Y[j];
                    r_matrix[i, j] = r[j] ? 1 : 0;
                }
            }

            CollaborativeFilteringRSCostFunction f = new CollaborativeFilteringRSCostFunction(Y_matrix, r_matrix, n_m, num_features, dimension);
            f.RegularizationLambda = mRegularizationLambda;
            double[] theta_x_0 = new double[dimension];
            for (int d = 0; d < dimension; ++d)
            {
                theta_x_0[d] = NextRandomDouble();
            }

            ContinuousSolution solution = mLocalSearcher.Minimize(theta_x_0, f, mMaxSolverIteration);

            mThetaX = solution.Values;

            mTheta = new DenseMatrix(n_u, num_features);
            for (int j = 0; j < n_u; ++j)
            {
                for (int k = 0; k < num_features; ++k)
                {
                    int index = j * num_features + k;
                    mTheta[j, k] = mThetaX[index];
                }
            }

            Matrix<double> X = new DenseMatrix(n_m, num_features);
            for (int i = 0; i < n_m; ++i)
            {
                T rec = data_set[i];
                if (rec.X == null)
                {
                    rec.X = new double[num_features];
                }
                for (int k = 0; k < num_features; ++k)
                {
                    int index = i * num_features + k + theta_dimension;
                    rec.X[k]=mThetaX[index];
                    X[i, k] = mThetaX[index];
                }
            }


            Matrix<double> XThetaPrime = X.Multiply(mTheta.Transpose());

            for (int i = 0; i < n_m; ++i)
            {
                T rec = data_set[i];
                for (int j = 0; j < n_u; ++j)
                {
                    if (r_matrix[i, j]==0)
                    {
                        rec.UserRanks[j] = XThetaPrime[i, j];
                    }
                }
            }

            return XThetaPrime;
            
        }

        public double ComputeCost(List<T> data_set, int num_features)
        {
            int n_m = data_set.Count;
            if (n_m == 0)
            {
                throw new ArgumentException("Data set is empty!");
            }
            int x_dimension = num_features;
            int n_u = data_set[0].UserRanks.Length;

            Matrix<double> Y_matrix = new SparseMatrix(n_m, n_u, 0);
            int[,] r_matrix = new int[n_m, n_u];

            int theta_dimension = n_u * x_dimension;
            int content_dimension = n_m * x_dimension;

            int dimension = theta_dimension + content_dimension;

            for (int i = 0; i < n_m; ++i)
            {
                T rating = data_set[i];
                double[] Y = rating.UserRanks;
                bool[] r = rating.IsRated;

                for (int j = 0; j < n_u; ++j)
                {
                    Y_matrix[i, j] = Y[j];
                    r_matrix[i, j] = r[j] ? 1 : 0;
                }
            }

            CollaborativeFilteringRSCostFunction f = new CollaborativeFilteringRSCostFunction(Y_matrix, r_matrix, n_m, x_dimension, dimension);
            f.RegularizationLambda = 0;

            return f.Evaluate(mThetaX);
        }
    }
}
