using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Recommender
{
    public class CollaborativeFilteringRS_ParamSelector<T>
        where T : RatedItem
    {
        public void Select_Lambda(List<T> X, List<T> Xval, int max_iterations, int num_features, out double best_lambda)
        {
            double[] lambda_candidates = new double[] { 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100 };

            int C_length = lambda_candidates.Length;

            double min_prediction_error = double.MaxValue;

            best_lambda = -1;

            for (int i = 0; i < C_length; ++i)
            {
                CollaborativeFilteringRS<T> svm = new CollaborativeFilteringRS<T>();
                svm.RegularizationLambda = lambda_candidates[i];
                svm.MaxSolverIteration = max_iterations;

                svm.Compute(X, num_features);

                double cost = svm.ComputeCost(Xval, num_features);

                if (min_prediction_error > cost)
                {
                    min_prediction_error = cost;
                    best_lambda = lambda_candidates[i];
                }
            }
        }
    }
}
