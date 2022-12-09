#import threading
import time
import numpy 
tm1=time.localtime(time.time())



#time.sleep(1)
tm2=time.localtime(time.time())
#calculate the time cost
import computeTime as ct
ct.getTimeCost(tm1,tm2)

