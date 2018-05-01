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

        public static void Main(string[] args)
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


    }
}

```
