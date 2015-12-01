# The minimal example that creates the error
import datajoint as dj

schema = dj.schema('sandbox',locals())

@schema
class Basic(dj.Manual):
    definition="""
    exp_date    :date   # primary key with type date
    ---
    path        :varchar(20)    # variable used in Dependent
    """

@schema
class Dependent(dj.Computed):
    definition="""
    -> Basic
    ---
    new_path    :varchar(21)    # variable computed
    """

    def _make_tuples(self,key):
        p = (Basic() & key).fetch1['path']

        new_path = p + 'new'

        self.insert1(key,new_path=new_path)
