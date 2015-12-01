# And how I run the file in an IPython Notebook
import datajoint as dj
import os

c = dj.conn()

%run sandbox.py

from sandbox import Basic
basic = Basic()
comp = Dependent()

basic.insert1({'exp_date':'2015-11-24','path':'example/path'})

comp.populate()

# For me it yields the following error:
# InternalError: (1630, "FUNCTION datetime.date does not exist. Check the 'Function Name Parsing and Resolution' section in the Reference Manual")
