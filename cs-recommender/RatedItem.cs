using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Recommender
{
    public class RatedItem 
    {
        protected double[] mX;
        protected double[] mY;
        protected bool[] mIsRated;

        protected int mItemIndex;

        public int ItemIndex
        {
            get { return mItemIndex; }
            set { mItemIndex = value; }
        }

        public RatedItem(double[] features, double[] ratings, bool[] is_rated)
        {
            mX = features;
            mY = ratings;
            mIsRated = is_rated;
        }

        public double[] X
        {
            get { return mX; }
            set { mX = value; }
        }

        public double[] UserRanks
        {
            get { return mY; }
        }

        public bool[] IsRated
        {
            get { return mIsRated; }
        }
    }
}
