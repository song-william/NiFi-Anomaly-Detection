from datetime import datetime
import random
sampleSize = 1000
distributions = []
for x in xrange(10):
    count = 0
    for i in xrange(sampleSize):
        t = datetime.now()
        if t.microsecond % 1000 == 0:
            count += 1
    distributions.append(float(count)/sampleSize)

print "percentage of anomalies", distributions

# time.sleep(random.random()*5)
