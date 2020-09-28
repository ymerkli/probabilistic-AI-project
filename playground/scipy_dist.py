import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np
from scipy.special import logsumexp

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])




def main():
    print(HYPOTHESIS_SPACE[0].logpdf(1))
    print(norm.logpdf(1, loc=0, scale=math.sqrt(VARIANCE)))


if __name__ == "__main__":
    main()
