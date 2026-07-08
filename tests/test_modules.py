import sys
import pandas as pd
import plotly
from docplex.cp.model import CpoModel

print("Python:", sys.version)
print("pandas:", pd.__version__)
print("plotly:", plotly.__version__)

mdl = CpoModel()
x = mdl.integer_var(0, 10, "x")
mdl.add(x >= 7)
sol = mdl.solve(TimeLimit=5)


"""
conda autostore
Python: 3.10.18 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:08:55) [MSC v.1929 64 bit (AMD64)]
pandas: 2.3.1
plotly: 6.0.1
 ! --------------------------------------------------- CP Optimizer 22.1.1.0 --
 ! Satisfiability problem - 1 variable, 1 constraint
 ! TimeLimit            = 5
 ! Initial process time : 0.01s (0.01s extraction + 0.00s propagation)
 !  . Log search space  : 2.0 (before), 2.0 (after)
 !  . Memory usage      : 266.8 kB (before), 266.8 kB (after)
 ! Using parallel search with 12 workers.
 ! ----------------------------------------------------------------------------
 !               Branches  Non-fixed    W       Branch decision
 *                      2  0.04s        1         8  = x
 ! ----------------------------------------------------------------------------
 ! Search completed, 1 solution found.
 ! ----------------------------------------------------------------------------
 ! Number of branches     : 78
 ! Number of fails        : 30
 ! Total memory usage     : 5.3 MB (5.2 MB CP Optimizer + 0.0 MB Concert)
 ! Time spent in solve    : 0.05s (0.04s engine + 0.01s extraction)
 ! Search speed (br. / s) : 2228.6
 ! ----------------------------------------------------------------------------
 
 optim conda autostore310
 Python: 3.10.18 | packaged by conda-forge | (main, Jun  4 2025, 14:45:41) [GCC 13.3.0]
pandas: 2.3.1
plotly: 6.0.0
 ! --------------------------------------------------- CP Optimizer 22.1.1.0 --
 
 
rci amdfast
ml --ignore-cache purge
ml CPLEX/22.1.0-foss-2022a
ml plotly.py/5.12.0-GCCcore-11.3.0

Python: 3.10.4 (main, Aug 12 2022, 16:49:44) [GCC 11.3.0]
pandas: 1.4.2
plotly: 5.12.0
 ! --------------------------------------------------- CP Optimizer 22.1.0.0 --

"""