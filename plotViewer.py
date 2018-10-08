#!/Users/cbacigalupo/canopy/bin/python

import matplotlib.pyplot as plt
import sys
import dill


try:
    ax = dill.load(open(sys.argv[1], "rb"))
    plt.show()
except:
    print('Not working')
    pass    