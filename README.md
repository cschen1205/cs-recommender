# cs-recommender

Recommender based on Hidden Factor Analysis Collaborative Filtering in .NET 4.6.1

# Install

Run the following command to get the nuget package:

```bash
Install-Package cs-recommender 
```

# Usage

The sample code show show how to train and use the Collaborative Filtering Recommender:

```cs 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using System.IO;
using ContinuousOptimization.LocalSearch;
using ContinuousOptimization;
using Recommender.Utils;

namespace Recommender
{
    public class Program
    {
        public static void Main(String[] args)
        {
            //Test_CollaborativeFilteringRSCostFunction_Evaluate();
            //Test_CollaborativeFilteringRSCostFunction_CalcGradient();
            Test_Compute();
        }

        protected static Matrix<double> Convert2Matrix(List<List<double>> X)
        {
            int row_count, col_count;
            DblDataTableUtil.GetSize(X, out row_count, out col_count);
            Matrix<double> X_matrix = new DenseMatrix(row_count, col_count, 0);

            for (int i = 0; i < row_count; ++i)
            {
                List<double> row = X[i];
                for (int j = 0; j < col_count; ++j)
                {
                    X_matrix[i, j] = row[j];
                }
            }

            return X_matrix;
        }

        protected static Random mRand = new Random();

        protected static Matrix<double> CreateRandomMatrix(int row_count, int col_count)
        {
            Matrix<double> X_t = new SparseMatrix(row_count, col_count);
            for (int i = 0; i < row_count; i++)
            {
                for (int j = 0; j < col_count; ++j)
                {
                    X_t[i, j] = mRand.NextDouble();
                }
            }

            return X_t;
        }

        protected static List<string> LoadMovies()
        {
            List<string> movie_titles = new List<string>();

            string line;
            using (StreamReader reader = new StreamReader("movie_ids.txt"))
            {
                while ((line = reader.ReadLine()) != null)
                {
                    string[] texts = line.Split(new char[] { ' ' });
                    StringBuilder sb = new StringBuilder();
                    bool first_index = true;
                    foreach (string text in texts)
                    {
                        if (string.IsNullOrEmpty(text)) continue;
                        if (first_index)
                        {
                            first_index = false;
                            continue;
                        }
                        sb.AppendFormat("{0} ", text);
                    }
                    string title = sb.ToString().Trim();
                    movie_titles.Add(title);
                }
            }

            return movie_titles;
        }

        public static void Test_Compute()
        {
            List<string> movie_titles = LoadMovies();
            int num_movies = movie_titles.Count;

            // Step 1: create my ratings with missing entries
            double[] my_ratings = new double[num_movies];
            int[] my_ratings_r = new int[num_movies];
            for (int i = 0; i < num_movies; ++i)
            {
                my_ratings[i] = 0;
            }

            my_ratings[1] = 4;
            my_ratings[98] = 2;
            my_ratings[7] = 3;
            my_ratings[12] = 5;
            my_ratings[54] = 4;
            my_ratings[64] = 5;
            my_ratings[66] = 3;
            my_ratings[69] = 5;
            my_ratings[183] = 4;
            my_ratings[226] = 5;
            my_ratings[355] = 5;

            for (int i = 0; i < num_movies; ++i)
            {
                my_ratings_r[i] = my_ratings[i] > 0 ? 1 : 0;
            }

            // Step 2: load the current ratings of all users, i.e., Y and R
            List<List<double>> Y = DblDataTableUtil.LoadDataSet("Y.txt");
            List<List<int>> R = IntDataTableUtil.LoadDataSet("R.txt");

            int num_users;
            DblDataTableUtil.GetSize(Y, out num_movies, out num_users);


            // Step 3: insert my ratings into the existing Y and R (as the first column)
            num_users++;
            List<RatedItem> records = new List<RatedItem>();
            for (int i = 0; i < num_movies; ++i)
            {
                double[] rec_Y = new double[num_users];
                bool[] rec_R = new bool[num_users];
                for (int j = 0; j < num_users; ++j)
                {
                    if (j == 0)
                    {
                        rec_Y[j] = my_ratings[i];
                        rec_R[j] = my_ratings_r[i] == 1;
                    }
                    else
                    {
                        rec_Y[j] = Y[i][j - 1];
                        rec_R[j] = R[i][j - 1] == 1;
                    }
                }
                RatedItem rec = new RatedItem(null, rec_Y, rec_R);
                records.Add(rec);
            }

            int num_features = 10;

            double lambda = 10;
            CollaborativeFilteringRS<RatedItem> algorithm = new CollaborativeFilteringRS<RatedItem>();
            algorithm.Stepped += (s, step) =>
            {
                Console.WriteLine("#{0}: {1}", step, s.Cost);
            };
            algorithm.RegularizationLambda = lambda;
            algorithm.MaxLocalSearchIteration = 100;
            GradientDescent local_search = algorithm.LocalSearch as GradientDescent;
            local_search.Alpha = 0.005;

            double[] Ymean;
            algorithm.DoMeanNormalization(records, out Ymean);

            algorithm.Compute(records, num_features);

            algorithm.UndoMeanNormalization(records, Ymean);

            int userId = 0;
            int topK = 10;
            List<RatedItem> highest_ranks = algorithm.SelectHigestRanked(userId, records, topK);

            for (int i = 0; i < highest_ranks.Count; ++i)
            {
                RatedItem rec = highest_ranks[i];
                Console.WriteLine("#{0}: ({1}) {2}", i + 1, rec.UserRanks[0], movie_titles[rec.ItemIndex]);
            }
        }

        public static void Test_CollaborativeFilteringRSCostFunction_CalcGradient(double lambda = 0)
        {

            Matrix<double> X_t = CreateRandomMatrix(4, 3);
            Matrix<double> Theta_t = CreateRandomMatrix(5, 3);

            Matrix<double> Y = X_t.Multiply(Theta_t.Transpose());
            int[,] R = new int[Y.RowCount, Y.ColumnCount];

            for (int i = 0; i < Y.RowCount; ++i)
            {
                for (int j = 0; j < Y.ColumnCount; ++j)
                {
                    if (mRand.NextDouble() > 0.5)
                    {
                        Y[i, j] = 0;
                    }
                    if (Y[i, j] == 0)
                    {
                        R[i, j] = 0;
                    }
                    else
                    {
                        R[i, j] = 1;
                    }
                }
            }


            Matrix<double> X = CreateRandomMatrix(4, 3);
            Matrix<double> Theta = CreateRandomMatrix(5, 3);
            int num_users = Y.ColumnCount;
            int num_movies = Y.RowCount;
            int num_features = Theta_t.ColumnCount;

            int dimension = num_movies * num_features + num_users * num_features; //total number of entries in X and Theta

            double[] theta_x = new double[dimension];
            CollaborativeFilteringRSCostFunction.UnrollMatrixIntoVector(Theta, X, theta_x);

            CollaborativeFilteringRSCostFunction f = new CollaborativeFilteringRSCostFunction(Y, R, num_movies, num_features, dimension);
            f.RegularizationLambda = lambda;

            double[] numgrad = new double[dimension];
            double[] grad = new double[dimension];
            GradientEstimation.CalcGradient(theta_x, numgrad, (x_pi, constraints) =>
            {
                return f.Evaluate(x_pi);
            });
            f.CalcGradient(theta_x, grad);

            Console.WriteLine("The relative difference will be small:");
            for (int i = 0; i < dimension; ++i)
            {
                Console.WriteLine("{0}\t{1}", numgrad[i], grad[i]);
            }
        }

        public static void Test_CollaborativeFilteringRSCostFunction_Evaluate(double lambda)
        {
            int num_users = 4; int num_movies = 5; int num_features = 3;

            List<List<double>> X = DblDataTableUtil.LoadDataSet("X.txt");
            List<List<double>> Y = DblDataTableUtil.LoadDataSet("Y.txt");
            List<List<int>> R = IntDataTableUtil.LoadDataSet("R.txt");
            List<List<double>> Theta = DblDataTableUtil.LoadDataSet("Theta.txt");

            X = DblDataTableUtil.SubMatrix(X, num_movies, num_features);
            Y = DblDataTableUtil.SubMatrix(Y, num_movies, num_users);
            R = IntDataTableUtil.SubMatrix(R, num_movies, num_users);
            Theta = DblDataTableUtil.SubMatrix(Theta, num_users, num_features);

            Matrix<double> Y_matrix = Convert2Matrix(Y);
            Matrix<double> X_matrix = Convert2Matrix(X);
            Matrix<double> Theta_matrix = Convert2Matrix(Theta);
            int[,] R_matrix = IntDataTableUtil.Convert2DArray(R);

            int dimension = num_movies * num_features + num_users * num_features; //total number of entries in X and Theta

            double[] theta_x = new double[dimension];
            CollaborativeFilteringRSCostFunction.UnrollMatrixIntoVector(Theta_matrix, X_matrix, theta_x);

            CollaborativeFilteringRSCostFunction f = new CollaborativeFilteringRSCostFunction(Y_matrix, R_matrix, num_movies, num_features, dimension);
            f.RegularizationLambda = lambda;
            double J = f.Evaluate(theta_x);

            Console.WriteLine("Cost at loaded parameters: {0} (this value should be about 22.22)", J);
        }

    }
}

```
