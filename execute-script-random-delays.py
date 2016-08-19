import time
from datetime import datetime
import random
flowFile = session.get()
if flowFile is None:
    exit()
t = datetime.now()
if t.microsecond % 1000 == 0:
    time.sleep(random.random()*5)
    flowFile = session.putAttribute(flowFile, 'anomaly', 'yes')

session.transfer(flowFile, REL_SUCCESS)
